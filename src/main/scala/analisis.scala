import java.util.Date
import config.Config
import service.DataBase
import service.RandomForestWithBalance
import service.Metrics
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
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors}
import org.apache.spark.mllib.stat.KernelDensity
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.tree.model.Split
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object analisis {

  def proRatio(row:org.apache.spark.sql.Row, inv:Boolean):Double ={
    val values=row.getAs[DenseVector](0)
    val small=0.00001
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
  def getDenText(dataIn:DataFrame,inText:String,ejeX:Array[Double],inv:Boolean):String={
    var coefLR: RDD[Double] = dataIn.rdd.map(row=>{proRatio(row,inv)})
    val x=ejeX
    val n=coefLR.count.toDouble
    val h=coefLR.stdev*scala.math.pow((4.0/3.0/n),1.0/5.0)
    val bw=0.1
    val densidad= denCal(coefLR,bw,x)
    val densidadTxt=for ((value, index) <- x.zipWithIndex)
      yield  (value, densidad(index))
    val out=densidadTxt.mkString(","+inText+"\n")+","+inText+"\n" filterNot ("()" contains _)
    out
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
  opt[String]('m', "mex").action( (x, c) =>
  c.copy(mex = x) ).text("memory executor (7g or 7000m)")
  opt[String]('h', "hmem").action( (x, c) =>
  c.copy(hmem = x) ).text("memory executor overhead (7g or 7000m)")
  opt[String]('e', "estrategia").action( (x, c) =>
  c.copy(estrategia = x) ).text("strategy to solve the imbalance(kmeans,meta,smote)")
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
  opt[String]('f', "filter").action( (x, c) =>
  c.copy(filter = x) ).text("filters of the tabla of input")
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
 	   val memex=config.mex
 	   val memover=config.hmem
 	   val est=config.estrategia
     val filtros=config.filter
     val ejesX=config.axes.toArray
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
     val conf = new SparkConf().setAppName("AnalisisP2P_balanced")
     .set("spark.executor.memory",memex)
     .set("spark.yarn.executor.memoryOverhead", memover)
     val sc = new SparkContext(conf)
     val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
     import sqlContext.implicits._
     // read the tabla base
     val db=new DataBase(tablaBase,numPartitions,sqlContext,filtros)
     // if opt==0 the table is readed otherwise the dataframe
     // is calculated from zero
     val labeledDF = db.getDataFrameLabeledLegalFraud(opt=="0").cache()
     var textOut="tipo,tp,fn,tn,fp,TPR,SPC,PPV,ACC,F1,MGEO,PEXC,MCC,areaRoc,impurity,depth,bins\n"
     var textImp="variable,importance,depth,impurity,bins\n"
     var textOut3=""
     var textRoc="X,Y,impurity,depth,bins\n"
     var txtDensidadAc="X,Y,type,depth,impurity,bins\n"
     var txtDensidad=""
     val featureIndexer = (new VectorIndexer()
     .setInputCol("features")
     .setOutputCol("indexedFeatures")
     .setMaxCategories(2)
     .fit(labeledDF))
    for ( a <- 1 to k)
    {
     for(params <- grid){
      logger.info("............using(impurity,depth, bins)............. " + params)
      val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.75, 0.25))

      // creation of the model
      var model = {
        if (est=="balanced") {
          new RandomForestWithBalance(trainingData,
            params,numPartitions,featureIndexer,
            Array(1.0,-1.0),sqlContext)
        } else{
        new RandomForestWithBalance(trainingData,
          params,numPartitions,featureIndexer,
          Array(1.0,-1.0),sqlContext)
        }
     }
     // training the model
     model.training()
     // getting the features importances
     val importances=model.featureImportances.toArray.map(_.toString)
     val ncol=db.getNamesCol
     val importancesName=for ((nam, index) <- ncol.zipWithIndex)
       yield  (nam, importances(index))
     val impSave=importancesName.mkString(","+params+"\n") + ","+params+"\n" filterNot ("()" contains _)
     val numTrees=model.getNumTrees
     textImp=textImp+(textImp)
     textOut3=(textOut3 + "---Learned classification tree ensemble model with"
      + params + ",trees="+ numTrees + "\n" + model.toDebugString + "\n")
     logger.info("..........Testing...............")
     // Make predictions.
     var predictions = model.getPredictions(testData,sc,"soft")
     logger.info("..........Calculate Error on test...............")
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
     textRoc=textRoc+met.roc.collect.mkString(","+params+"\n")+" "+params+"\n" filterNot ("()" contains _)
     val aROC=met.areaUnderROC
     textOut=(textOut + "test," + tp + "," + fn + "," + tn + "," + fp + "," +
       TPR + "," + SPC + "," + PPV + "," + acc + "," + f1  +  "," +mGeo +  ","
        + pExc + "," + MCC + "," + aROC + "," + params + "\n"  filterNot ("()" contains _) )
     predictions.unpersist()
     logger.info("..........Calculate Error on Training...............")
     // Make predictions.
     predictions = model.getPredictions(trainingData,sc)
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
    logger.info("..........writing the files...............")
    val pw = new PrintWriter(new File(salida+"Confusion.csv" ))
    pw.write(textOut)
    pw.close
    val pw2 = new PrintWriter(new File(salida+"Importances.csv" ))
    pw2.write(textImp)
    pw2.close
    val pw3 = new PrintWriter(new File(salida+"Model.csv" ))
    pw3.write(textOut3)
    pw3.close
    val pw4 = new PrintWriter(new File(salida+"Roc.csv" ))
    pw4.write(textRoc)
    pw4.close
    // define the x axis
    val axis=(ejesX(0).toDouble to ejesX(1).toDouble by 0.5d).toArray
    logger.info("..........getting densidity...............")
    val predLegal = predictions.where("label=-1.0")
    var predDen = predLegal.select("probability")
    logger.info("..........getting densidity legal...............")
    val d1= getDenText(predDen,"Legal," +params,axis,false)
    val predFraud= predictions.where("label=1.0")
    predDen = predFraud.select("probability")
    logger.info("..........getting densidity fraude...............")
    val d2= getDenText(predDen,"Fraude," +params,axis,false)
    txtDensidad=d1+d2
    txtDensidadAc=txtDensidadAc+txtDensidad
    val pwdensidad = new PrintWriter(new File(salida+"_denisad.csv" ))
    pwdensidad.write(txtDensidadAc)
    pwdensidad.close
    predictions.unpersist()
    logger.info("..........termino..............")
}

 //

}
logger.info("..........termino programa..............")
sc.stop()
case None =>
    println(".........arguments are bad...............")
}

  }
}






/*Aqui!!!!!!!!!!!!!!!!!!!!!!


nohup spark-submit --driver-memory 10g --num-executors 7 --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i variables_finales_tarjeta -p 100 -h 1000 -m 9000m -r 0 -o balanced_test -e balanced -k 4 -i gini,entropy -d 10,20,30 -b 32,128 -a -50,50 > balanced_test_log 2>&1&

nohup spark-submit --driver-memory 10g --num-executors 5 --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i test -p 100 -h 1000 -m 9000m -r 0 -o test -e balanced -k 4 -i gini,entropy -d 20,30 -b 32,128 -a -25,25 > test 2>&1&



spark-submit --verbose --driver-memory 10g --class "analisis" AnalisisP2P_Balanced-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 0 -o balanced -e balanced -k 4 -i gini,entropy -d 10,20,30 -b 100 -a -25,25
*/
