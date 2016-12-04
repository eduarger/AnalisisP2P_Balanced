import java.util.Date
import config.Config
import service.DataBase
import java.io._
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.clustering.KMeans
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



object analisis {

  def proRatio(row:org.apache.spark.sql.Row, inv:Boolean):Double ={
    val values=row.getAs[DenseVector](0)
    val log = {
      if(inv)
        math.log(values(1)/values(0))
      else
        math.log(values(0)/values(1))
    }
    log
  }

  def denCal(sample: RDD[Double], bw:Array[Double], x:Array[Double]):Array[Double] ={
    var densities=Array[Double]()
    for (b <- bw )
    {
      val kd = (new KernelDensity()
      .setSample(sample)
      .setBandwidth(b))
      densities = kd.estimate(x)
  }
    densities
  }
// tiene que entrear un dataframe con la probilidad, similar al rwa predcition
  def getDenText(dataIn:DataFrame,inText:String,ejeX:Array[Double],inv:Boolean):String={
    var out=inText+","
    var coefLR: RDD[Double] = dataIn.rdd.map(row=>{proRatio(row,inv)})
    val x=ejeX
    val bw= Array(2.0)
    val densidad= denCal(coefLR,bw,x)
    out=out+densidad.mkString(", ") + "\n"
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
     val logger= LogManager.getLogger("analisis")
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
     val conf = new SparkConf().setAppName("AnalisisP2P")
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
     var textOut="tipo,tp,fn,tn,fp,TPR,SPC,PPV,ACC,F1,MGEO,PEXC,MCC,impurity,depth,bins\n"
     var textOut2=""
     var textOut3=""
     var txtDendsidad=""
     var txtDendsidadAc=""
     val featureIndexer = (new VectorIndexer()
     .setInputCol("features")
     .setOutputCol("indexedFeatures")
     .setMaxCategories(2)
     .fit(labeledDF))
    for (params <- grid )
    {
     for( a <- 1 to k){
      logger.info("..........inicinando ciclo con un valor de trees..............."+ params._1)
      println("using(trees,impurity,depth, bins) " + params)
      val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))

      // creation of the model
      var model = {
        if (est=="balanced") {
          new RandomForestWithBalance(trainingData,
            params,numPartitions,featureIndexer,
            Array(1.0,-1.0),sqlContext,sc)
        } else{
        new RandomForestWithBalance(trainingData,
          params,numPartitions,featureIndexer,
          Array(1.0,-1.0),sqlContext,sc)
        }
     }
     // training the model
     model.training()
     // getting the features importances
     val importances=model.featureImportances.toArray
     val numTrees=model.getNumTrees
     //
     textOut2=(textOut2 + params + "," + importances.mkString(", ") + "\n" )
     textOut3=(textOut3 + "---Learned classification tree ensemble model with"
      + params + ",trees="+ numTrees + "\n" + model.toDebugString + "\n")
     logger.info("..........Testing...............")
     // Make predictions.
     var predictions = model.getPredictions(testData)
     logger.info("..........Calculate Error on test...............")
     predictions.persist()
     var predRow: RDD[Row]=predictions.select("label", "predictedLabel").rdd
     var predRDD: RDD[(Double, Double)] = predRow.map(row=>{(row.getDouble(0), row.getDouble(1))})
     var tp=predRDD.filter(r=>r._1== 1.0 && r._2==1.0).count().toDouble
     var fn=predRDD.filter(r=>r._1== 1.0 && r._2== -1.0).count().toDouble
     var tn=predRDD.filter(r=>r._1== -1.0 && r._2== -1.0).count().toDouble
     var fp=predRDD.filter(r=>r._1== -1.0 && r._2== 1.0).count().toDouble
     var TPR = (tp/(tp+fn))*100.0
     var SPC = (tn/(fp+tn))*100.0
     var PPV= (tp/(tp+fp))*100.0
     var acc= ((tp+tn)/(tp+fn+fp+tn))*100.0
     var f1= ((2*tp)/(2*tp+fp+fn))*100.0
     var mGeo=math.sqrt(TPR*SPC)
     var pExc=(tp*tn-fn*fp)/((fn+tp)*(tn+fp))
     var MCC=(tp*tn-fp*fn)/math.sqrt((fn+tp)*(tn+fp)*(fp+tp)*(fn+tn))
     textOut=(textOut + "test,"  + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
       PPV + "," + acc + "," + f1  +  "," +mGeo +  "," + pExc + "," + MCC + "," + params + "\n" )
     predictions.unpersist()
     logger.info("..........Calculate Error on Training...............")
     // Make predictions.
     predictions = model.getPredictions(testData)
     predictions.persist()
     //TODO: define a Class for the metrics
     predRow = predictions.select("label", "predictedLabel").rdd
     predRDD = (predRow.map(row=>{(row.getDouble(0), row.getString(1).toDouble)}))
     tp=predRDD.filter(r=>r._1== 1.0 && r._2==1.0).count().toDouble
     fn=predRDD.filter(r=>r._1== 1.0 && r._2== -1.0).count().toDouble
     tn=predRDD.filter(r=>r._1== -1.0 && r._2== -1.0).count().toDouble
     fp=predRDD.filter(r=>r._1== -1.0 && r._2== 1.0).count().toDouble
     TPR = (tp/(tp+fn))*100.0
     SPC = (tn/(fp+tn))*100.0
     PPV= (tp/(tp+fp))*100.0
     acc= ((tp+tn)/(tp+fn+fp+tn))*100.0
     f1= ((2*tp)/(2*tp+fp+fn))*100.0
     mGeo=math.sqrt(TPR*SPC)
     pExc=(tp*tn-fn*fp)/((fn+tp)*(tn+fp))
     MCC=(tp*tn-fp*fn)/math.sqrt((fn+tp)*(tn+fp)*(fp+tp)*(fn+tn))
     textOut=(textOut + "train," +  params._1 + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
       PPV + "," + acc + "," + f1  +  "," +mGeo +  "," + pExc + "," + MCC + "," + params + "\n" )
     predictions.unpersist()
    logger.info("..........writing the files...............")
    val pw = new PrintWriter(new File(salida+"Confusion.txt" ))
    pw.write(textOut)
    pw.close
    val pw2 = new PrintWriter(new File(salida+"Importances.txt" ))
    pw2.write(textOut2)
    pw2.close
    val pw3 = new PrintWriter(new File(salida+"Model.txt" ))
    pw3.write(textOut3)
    pw3.close
    // define the x axis
    val axis=(ejesX(0).toDouble to ejesX(1).toDouble by 0.5d).toArray
    logger.info("..........getting densidity...............")
    val predLegal = predictions.where("label=-1.0")
    var predDen = predLegal.select("probability")
    logger.info("..........getting densidity legal...............")
    val d1= getDenText(predDen,"Legal," + ","+params,axis,true)
    val predFraud= predictions.where("label=1.0")
    predDen = predFraud.select("probability")
    logger.info("..........getting densidity fraude...............")
    val d2= getDenText(predDen,"Fraude,"+params._1+ ","+params,axis,true)
    txtDendsidad="clase,imp,depth,bines,"+axis.mkString(", ") + "\n"+d1+d2
    txtDendsidadAc=txtDendsidadAc+ "\n"+ txtDendsidad
    val pwdensidad = new PrintWriter(new File(salida+"_denisad.csv" ))
    pwdensidad.write(txtDendsidadAc)
    pwdensidad.close
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



spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i base_tarjeta -p 500 -h 1000 -m 13500m -r 1 -o testrf1 -e rfpure -k 1 -t 50 -i gini,entropy -d 30 -b 72

spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i base_tarjeta -p 100 -h 1000 -m 13500m -r 1 -o testrf1 -e rfpure -k 1 -t 50 -i gini -d 30 -b 32
spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i base_tarjeta_complete -p 100 -h 1000 -m 13500m -r 1 -o testrf1 -e rfpure -k 1 -t 25 -i gini -d 30 -b 32

spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 13500m -r 1 -o rf1 -e rfpure -k 5 -t 1,10,25,50,100 -i gini,entropy -d 10,20,32 -b 32,72

spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 13500m -r 1 -o test1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72


spark-submit --master yarn --deploy-mode cluster --driver-memory 1g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 1 -o p1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72

spark-submit --driver-memory 1g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 1 -o p1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72

nohup spark-submit a.py > archivo_salida 2>&1&
nohup spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 1 -o p1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72 -a -30,30 > archivo_salida 2>&1&


nohup spark-submit --driver-memory 6g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 0 -o p1 -e rfpure -k 4 -t 25,50,100 -i gini,entropy -d 10,20,30 -b 32,72 -a -25,25 > archivo_salida 2>&1&
nohup spark-submit --driver-memory 10g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 12000m -r 0 -o p1 -e rfpure -k 4 -t 100 -i gini,entropy -d 10,20,30 -b 32,72 -a -25,25 > archivo_salida 2>&1&

nohup spark-submit --num-executors 2 --driver-memory 10g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 12000m -r 0 -o p1 -e rfpure -k 4 -t 2 -i gini,entropy -d 10,20,30 -b 32,72 -a -25,25 > archivo_salida 2>&1&

nohup spark-submit --num-executors 3 --driver-memory 10g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 12000m -r 0 -o pbinhigh -e rfpure -k 4 -t 1,2,25,100 -i gini,entropy -d 30 -b 128,384 -a -25,25 > archivo_salida 2>&1&



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

  opt[Seq[Int]]('t', "trees").valueName("<trees1>,<trees1>...").action( (x,c) =>
    c.copy(trees = x) ).text("trees to evaluate")

  opt[Seq[String]]('i', "imp").valueName("<impurity>,<impurity>...").action( (x,c) =>
    c.copy(imp = x) ).text("impurity to evaluate")

  opt[Seq[Int]]('d', "depth").valueName("<depth1>,<depth2>...").action( (x,c) =>
    c.copy(depth = x) ).text("depth to evaluate")

  opt[Seq[Int]]('b', "bins").valueName("<bins1>,<bins2>...").action( (x,c) =>
    c.copy(bins = x) ).text("bins to evaluate")

  help("help").text("prints this usage text")

  libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"
--repositories "com.github.fommil.netlib" % "all" % "1.1.2"

spark-shell --driver-memory 4g --executor-memory 8g











*/
