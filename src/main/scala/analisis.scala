import java.util.Date
import config.Config
import service.DataBase
import service.RandomForestWithBalance
import service.Metrics
import service.EER
import java.io._
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{SparseVector, DenseVector,Vectors}
import org.apache.spark.mllib.stat.KernelDensity
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.tree.model.Split
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession

object analisis {

  def proRatio(row:org.apache.spark.sql.Row, inv:Boolean):Double ={
    val values=row.getAs[DenseVector](0)
    val small=0.0000000000001
    val log = {
      if(inv&&values(1)!=0&&values(0)!=0.0)
        math.log(values(1)/values(0))
      else if(inv&&values(1)==0&&values(0)!=0.0)
        math.log(small/values(0))
      else if(inv&&values(1)!=0&&values(0)==0.0)
          math.log(values(1)/small)
      else if(!inv&&values(1)!=0&&values(0)!=0.0)
        math.log(values(0)/values(1))
      else if(!inv&&values(1)==0&&values(0)!=0.0)
        math.log(values(0)/small)
      else if(!inv&&values(1)!=0&&values(0)==0.0)
        math.log(small/values(1))
      else
        math.log(values(1)/values(0))
    }
    log
  }

  def denCal(sample: RDD[Double], bw:Double, x:Array[Double]):Array[Double] ={
    var densities=Array[Double]()
      val kd = (new KernelDensity()
      .setSample(sample)
      .setBandwidth(bw))
      densities = kd.estimate(x)
      densities
  }



  // tiene que entrear un dataframe con la probilidad, similar al rwa predcition
  def getDenText(dataIn:DataFrame,inText:String,ejeX:Array[Double],inv:Boolean, k:Int):String={
    var coefLR: RDD[Double] = dataIn.rdd.map(row=>{proRatio(row,inv)})
    val x=ejeX
    val n=coefLR.count.toDouble
    val h=coefLR.stdev*scala.math.pow((4.0/3.0/n),1.0/5.0)
    val bw=0.1
    val densidad= denCal(coefLR,h,x)
    val densidadTxt=for ((value, index) <- x.zipWithIndex)
      yield  (value, densidad(index))
    val out=densidadTxt.mkString(","+k+","+inText+"\n")+","+k+","+inText+"\n" filterNot ("()" contains _)
    out
      }

    // function to save a string in specific file
    def saveTxtToFile(save:String, file:String): Unit ={
      val f = new File (file)
      f.getParentFile().mkdirs()
      val writer = new PrintWriter(f)
      writer.write(save)
      writer.close
    }


def main(args: Array[String]) {

  val parser = new scopt.OptionParser[Config]("scopt") {
  head("Analisis P2P", "0.1")
  opt[String]('i', "in").action( (x, c) =>
  c.copy(in = x) ).text("base table")
  opt[Int]('p', "par").action( (x, c) =>
  c.copy(par = x) ).text("par is an integer of num of partions")
  opt[String]('r', "read").action( (x, c) =>
  c.copy(read = x) ).text("read is parameter that says wich is the base table")
  opt[String]('o', "out").action( (x, c) =>
  c.copy(out = x) ).text("nameof the outfiles")
  opt[Int]('k', "kfolds").action( (x, c) =>
  c.copy(kfolds = x) ).text("kfolds is an integer of num of folds")
  opt[Seq[String]]('i', "imp").valueName("<impurity>,<impurity>...").action( (x,c) =>
  c.copy(imp = x) ).text("impurity to evaluate")
  opt[Seq[Int]]('d', "depth").valueName("<depth1>,<depth2>...").action( (x,c) =>
  c.copy(depth = x) ).text("depth to evaluate")
  opt[Seq[Int]]('b', "bins").valueName("<bins1>,<bins2>...").action( (x,c) =>
  c.copy(bins = x) ).text("bins to evaluate")
  opt[Seq[Int]]('a', "axis").valueName("<start>,<end>").action( (x,c) =>
  c.copy(axes = x) ).text("range of axis of the densidity")
  opt[Double]('t', "train").action( (x, c) =>
  c.copy(train = x) ).text("percentaje of sampel to train the system")
  opt[String]('f', "filter").action( (x, c) =>
  c.copy(filter = x) ).text("filters of the tabla of input")
  opt[Boolean]('D', "densidad").valueName("if the densidity and other measure is calculated").action( (x,c) =>
  c.copy(densidad = x) ).text("depth to evaluate")
  help("help").text("prints this usage text")
}
// parser.parse returns Option[C]
  parser.parse(args, Config()) match {
  case Some(config) =>
     val logger= LogManager.getLogger("AnalisisP2P_balanced")
     logger.setLevel(Level.INFO)
     logger.setLevel(Level.DEBUG)
     Logger.getLogger("org").setLevel(Level.WARN)
     Logger.getLogger("hive").setLevel(Level.WARN)
     logger.info("........getting the parameters...............")
     //lectura de parametros
     val tablaBase=config.in
     val numPartitions=config.par
 	   val k=config.kfolds
 	   val arrayNDepth=config.depth.toArray
 	   val arrayBins=config.bins.toArray
 	   val opt=config.read
 	   val salida=config.out
 	   val imp=config.imp.toArray
     val filtros=config.filter
     val ejesX=config.axes.toArray
     val pTrain=config.train
     val denFlag=config.densidad
     logger.info("Taking the folliwng filters: "+ filtros)
     logger.info("..........buliding grid of parameters...............")
     val grid = for {
           x <- imp
           y <- arrayNDepth
           z <- arrayBins
     } yield(x,y,z)
     // printing the grid
     logger.info("..........Here the grid constructed...............")
     for (a <- grid) println(a)
     //Begin the analysis
     logger.info("Solicitando recursos a Spark")
     val spark = SparkSession
       .builder()
       .appName("AnalisisP2P_Balnced2.1")
       .enableHiveSupport()
       .getOrCreate()
     val conf = new SparkConf().setAppName("AnalisisP2P")
     val sqlContext = spark.sqlContext
     // read the tabla base
     val db=new DataBase(tablaBase,numPartitions,sqlContext,filtros)
     // if opt==0 the table is readed otherwise the dataframe
     // is calculated from zero
     val labeledDF = db.getDataFrameLabeledLegalFraud(opt=="0",spark).cache()
     val ncol=db.getNamesCol
     var textOut="tipo,tp,fn,tn,fp,TPR,SPC,PPV,ACC,F1,MGEO,PEXC,MCC,areaRoc,impurity,depth,bins\n"
     var textImp="variable,importance,impurity,depth,bins\n"
     var textOut3=""
     var textRoc="X,Y,k,impurity,depth,bins\n"
     var txtDensidadAc="X,Y,k,type,impurity,depth,bins\n"
     var txtDensidad=""
     var txtDensidadAc2="X,Y,k,type,impurity,depth,bins\n"
     var txtDensidad2=""
     var txtEER="k,type,EER,LR,state,impurity,depth,bins\n"
     var txtHist="X,Y,k,type,impurity,depth,bins\n"
     val featureIndexer = (new VectorIndexer()
     .setInputCol("features")
     .setOutputCol("indexedFeatures")
     .setMaxCategories(2)
     .fit(labeledDF))
     //transforming all the data
     val dfTemporal= featureIndexer.transform(labeledDF).select("label", "indexedFeatures")
     logger.info("........writing "+ salida +  "_temporal..............")
     dfTemporal.write.mode(SaveMode.Overwrite).saveAsTable(salida+"_temporal")
     labeledDF.unpersist()
     logger.info("........reading "+ salida +  "_temporal..............")
     val dataTransformed=sqlContext.sql("SELECT * FROM "+salida+"_temporal").coalesce(numPartitions).cache()
    for ( a <- 1 to k)
    {
     for(params <- grid){
      val salidaDir="/home/german.melo/files/"+salida+"/"+params._1+params._2+params._3+"_"+a+"/out"
      logger.info("............using(impurity,depth, bins)............. " + params)
      logger.info("............using percetanje for train: " + pTrain +" and testing: " + (1.0-pTrain))
      val Array(trainingData, testData) = dataTransformed.randomSplit(Array(pTrain, 1.0-pTrain))

      // creation of the model
      var model =new RandomForestWithBalance(trainingData,
            params,numPartitions,featureIndexer,
            Array(1.0,-1.0),sqlContext)
     // training the model
     model.training()
     // getting the features importances
    logger.info("..........Testing...............")
    // define the x axis
     val axis=(ejesX(0).toDouble to ejesX(1).toDouble by 0.5d).toArray
     // Make predictions.
     logger.info("..........Calculate Error on test...............")
     var predictions = model.getPredictions(testData,spark,"soft")
     predictions.persist()
     val testMetrics=new Metrics(predictions, Array(1.0,-1.0))
     var tp=testMetrics.tp
     var fn=testMetrics.fn
     var tn=testMetrics.tn
     var fp=testMetrics.fp
     var TPR = testMetrics.sens
     var SPC = testMetrics.spc
     var PPV= testMetrics.pre
     var acc= testMetrics.acc
     var f1= testMetrics.f1
     var mGeo=testMetrics.mGeo
     var pExc=testMetrics.pExc
     var MCC=testMetrics.MCC
    // ROC metrics
     val met = new BinaryClassificationMetrics(predictions.select("Predictedlabel", "label").rdd.map(row=>(row.getDouble(0), row.getDouble(1))))
     // calling the EER class
     val eer = new EER()
     textRoc=textRoc+met.roc.collect.mkString(","+a+","+params+"\n")+","+a+","+params+"\n" filterNot ("()" contains _)
     val aROC=met.areaUnderROC
     textOut=(textOut + "test," + tp + "," + fn + "," + tn + "," + fp + "," +
       TPR + "," + SPC + "," + PPV + "," + acc + "," + f1  +  "," +mGeo +  ","
        + pExc + "," + MCC + "," + aROC + "," + params + "\n"  filterNot ("()" contains _) )


      var predLegal = predictions.where("label=-1.0")
      var predFraud= predictions.where("label=1.0")
       if(denFlag){
         logger.info("..........getting densidity for testing...............")
         var predDen = predLegal.select("probability")
         var d1= getDenText(predDen,"Legal," +params,axis,true,a)
         predDen = predFraud.select("probability")
         logger.info("..........getting densidity fraude...............")
         var d2= getDenText(predDen,"Fraude," +params,axis,true,a)
         txtDensidad=d1+d2
         txtDensidadAc=txtDensidadAc+txtDensidad
         // saving into file
         saveTxtToFile(txtDensidadAc,salidaDir+"_denisad_test.csv")
         logger.info("..........getting EER for test...............")
         val eerTest=eer.compute(predFraud,predLegal,0.00001)
         txtEER=txtEER + a + "," +"test,"+ eerTest + "," + params + "\n" filterNot ("()[]" contains _)
         // saving into file
         saveTxtToFile(txtEER,salidaDir+"_EER.csv")
         logger.info("..........getting EER plots...............")
         var txtEERPlot=eer.computePlots(predFraud,predLegal,params,a,150, "test", spark)
         txtHist=txtHist+txtEERPlot
         // saving into file
         saveTxtToFile(txtHist,salidaDir+"_EER_Plots.csv")
       }

     logger.info("..........saving testScores...............")
     eer.saveScores(predFraud,predLegal,"scoresTest/"+salida+"/"+params._1+params._2+params._3+"_"+a,spark)
     predictions.unpersist()
     // training data
     logger.info("..........Calculate Error on Training...............")
     // Make predictions.
     predictions = model.getPredictions(trainingData,spark,"soft")
     predictions.persist()
     //TODO: define a Class for the metrics
     val trainMetrics=new Metrics(predictions, Array(1.0,-1.0))
     tp=trainMetrics.tp
     fn=trainMetrics.fn
     tn=trainMetrics.tn
     fp=trainMetrics.fp
     TPR = trainMetrics.sens
     SPC = trainMetrics.spc
     PPV= trainMetrics.pre
     acc= trainMetrics.acc
     f1= trainMetrics.f1
     mGeo=trainMetrics.mGeo
     pExc=trainMetrics.pExc
     MCC=trainMetrics.MCC
     textOut=(textOut + "train," + tp + "," + fn + "," + tn + "," + fp + "," +
       TPR + "," + SPC + "," + PPV + "," + acc + "," + f1  +  "," +mGeo +  ","
        + pExc + "," + MCC + "," + aROC + "," + params + "\n"  filterNot ("()" contains _) )
        predLegal = predictions.where("label=-1.0")
        predFraud= predictions.where("label=1.0")
        if(denFlag){
          logger.info("..........getting densidity for training...............")
          var predDen = predLegal.select("probability")
          logger.info("..........getting densidity legal...............")
          var d1= getDenText(predDen,"Legal," +params,axis,true,a)
          predDen = predFraud.select("probability")
          logger.info("..........getting densidity fraude...............")
          var d2= getDenText(predDen,"Fraude," +params,axis,true,a)
          txtDensidad2=d1+d2
          txtDensidadAc2=txtDensidadAc2+txtDensidad2
          // saving into file
          saveTxtToFile(txtDensidadAc2,salidaDir+"_denisad_train.csv")
          logger.info("..........getting EER for train...............")
          val eerTrain=eer.compute(predFraud,predLegal,0.00001)
          txtEER=txtEER + a + "," +"train,"+ eerTrain + "," + params + "\n" filterNot ("()[]" contains _)
          // saving into file
          saveTxtToFile(txtEER,salidaDir+"_EER.csv")
          logger.info("..........getting EER plots...............")
          var txtEERPlot=eer.computePlots(predFraud,predLegal,params,a,150, "train", spark)
          txtHist=txtHist+txtEERPlot
          // saving into file
          saveTxtToFile(txtHist,salidaDir+"_EER_Plots.csv")
        }
        logger.info("..........saving train Scores...............")
        eer.saveScores(predFraud,predLegal,"scoresTrain/"+salida+"/"+params._1+params._2+params._3+"_"+a, spark)
    predictions.unpersist()
    logger.info("..........writing the files...............")
    val importances=model.featureImportances.toArray.map(_.toString)
    val importancesName=for ((nam, index) <- ncol.zipWithIndex)
      yield  (nam, importances(index))
    val impSave=importancesName.mkString(","+params+"\n") + ","+params+"\n" filterNot ("()" contains _)
    val numTrees=model.getNumTrees
    textImp=textImp+(impSave)
    textOut3=(textOut3 + "---Learned classification tree ensemble model with"
     + params + ",trees="+ numTrees + "\n" + model.toDebugString + "\n")
    // saving into file
    saveTxtToFile(textOut,salidaDir+"Confusion.csv")
    // saving into file
    saveTxtToFile(textImp,salidaDir+"Importances.csv")
    // saving into file
    saveTxtToFile(textOut3,salidaDir+"Model.txt")
    // saving into file
    saveTxtToFile(textRoc,salidaDir+"Roc.csv")
    model.saveModel("modelos/"+salida+"/"+params._1+params._2+params._3+"_"+a,spark.sparkContext)
    logger.info("..........termino..............")
}
//
}
logger.info("..........termino programa..............")
spark.stop()
case None =>
    println(".........arguments are bad...............")
}

  }
}
/*Aqui!!!!!!!!!!!!!!!!!!!!!!


nohup spark-submit --driver-memory 10g --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i variables_finales_tarjeta -p 96 -h 1000 -m 11000m -r 1 -o balanced_13Ene -e balanced -k 5 -i gini,entropy -d 30,10 -b 32,128,256 -a -50,50 -t 0.7 -f "monto<100000000" > balanced_13Ene_log 2>&1&

nohup spark-submit --driver-memory 10g --num-executors 5 --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i test -p 100 -h 1000 -m 9000m -r 0 -o test -e balanced -k 4 -i gini,entropy -d 20,30 -b 32,128 -a -25,25 > test 2>&1&


spark-submit --driver-memory 10g --num-executors 5 --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i test -p 100 -h 1000 -m 9000m -r 0 -o test -e balanced -k 2 -i gini -d 10 -b 20 -a -25,25


spark-submit --verbose --driver-memory 10g --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 0 -o balanced -e balanced -k 4 -i gini,entropy -d 10,20,30 -b 100 -a -25,25
*/
