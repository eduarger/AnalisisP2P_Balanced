package service
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.util.collection.OpenHashMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{types, _}

/**
 * Clase para que define un clasificador basado en Random Forest
 * Cada random forest tendra un solo arbol pero sera entrenado
 * como un subconjunto balanceado
 * Para iniciar tiene los siguentes parametros:
 * baseTrain: dataframe con labeledpoint
 * parameters=(impurity,depth, bins)
 * numP: numero de particiones
 * sqlContext: contexto de SQL, sc: spark context
 * laIndx: feIndx:  laConv:
 *
*/
class RandomForestWithBalance(
    baseTrain: DataFrame,
    parameters: (String, Int, Int),
    numP: Int,
    feIndx: VectorIndexerModel,
    labels: Array[Double],
    sqlContext: SQLContext,
    sc: SparkContext)
    extends Serializable {

   val numPartitions=numP
   private var metaClassifier = scala.collection.mutable.ArrayBuffer.empty[RandomForestModel]//[PipelineModel]
   // training dataset empty, in this dataset will be saved the balanced dataset
   private var trainDataset=sqlContext.createDataFrame(sc.emptyRDD[Row], baseTrain.schema)
   // relation for balance the training datasets
   val numOfMinorClass= baseTrain.where("label="+labelMinor).count()
   val numOfMayorClass= baseTrain.where("label!="+labelMinor).count()
   // calculate the smaple fraction and the number of classifiers
   private val sampleFraction=numOfMinorClass.toDouble/numOfMayorClass.toDouble
   private val fractionInv=numOfMayorClass/numOfMinorClass
   // avoid odd values
   val private numClassifiers=if (fractionInv%2==0) fractionInv + 1 else fractionInv
   // indexers and converters
   private val featureIndex=feIndx
   //labeles
   private val labelMinor=labels(0)
   private val labelMayor=labels(1)
   @transient private val logger = LogManager.getLogger("RandomForestWithBalance")
   logger.info("Number of classifiers(trees): "+numClassifiers)
   logger.info("sample rate is: "+sampleFraction)

  //set the training dataset
  def getTrainingBalancedSet() : Unit ={
    val dfMinor= baseTrain.where("label="+labelMinor)
    val dfMayor= baseTrain.where("label!="+labelMinor).sample(false,sampleFraction)
    trainDataset=dfMinor.unionAll(dfMayor)
   }
  //Convert dataframe to RDD with labeled parsed
  private def toRDDLabeledParsed(dataInput:DataFrame):RDD[LabeledPoint]={
    val rows=dataInput.rdd
    val labeledData: RDD[LabeledPoint]=rows.map(row =>{
      // convert the labels if is equal to labelMinor so 1.0 for other 0.0
      val newLabel= if(row.getDouble(0)==labelMinor) 1.0 else 0.0
      val res=LabeledPoint(newLabel,row.getAs[SparseVector](1))
      res})
    labeledData
  }

  //training one tree
  def private trainingTree() : RandomForestModel  = {
    // set the training dataset in each traiing
    getTrainingBalancedSet()
    val labeledPoints: RDD[LabeledPoint]=toRDDLabeledParsed(trainDataset)
    val model = RandomForest.trainClassifier(labeledPoints, 2, cat,
    1, "auto", parameters._1, parameters._2, parameters._3)
    model
    }

  // training all
  def training() : Unit  = {
    logger.info("..........Training...............")
    //cache the baseTrain
    baseTrain.persist()
    for( i <- 1 to numClassifiers.toInt){
      metaClassifier += trainingTree()
      if(i%10==0) logger.info("Acabo con clasificardor:"+i)
    }
    //unpersist the baseTrain
    baseTrain.persist()
    logger.info("..........Training Finished!...............")
  }

  // predcit one sample of a RDD
  private def predictOneSample(sample: LabeledPoint,
    models: scala.collection.mutable.ArrayBuffer[RandomForestModel]) = {
     val mapResults=metaClassifier.map(mod => mod.predict(sample.features))
     val proClaseUno=mapResults.filter(_==1).size.toDouble/numClassifiers
     val proClaseCero=mapResults.filterNot(_==1).size.toDouble/numClassifiers
     val predLabel=if(proClaseUno>proClaseCero) labelMinor else labelMayor
     val originalLabel=if(sample.label==1) labelMinor else labelMayor
     (originalLabel,sample.features,Array(proClaseCero,proClaseUno), predLabel)
   }
  // predict the result of one dataframe
  def getPredictions(testData:DataFrame) : DataFrame ={
    val labeledPoints: RDD[LabeledPoint]=toRDDLabeledParsed(testData)
    //broadcast all the trees trained
    val treesBroadcasted=sc.broadcast(metaClassifier)
    val rddPredictions=labeledPoints.map(point=>{
    predictOneSample(point,treesBroadcasted.value)})
    //convert to dataframe
    import sqlContext.implicits._
    val dataFramePredcited=rddPredictions.toDF("label", "features","probability","predictedLabel")
    dataFramePredcited
  }
  /**
   * Method to export to a comprenhisive text all the trees trained
   */
  def debugToString():String ={
    val texto=metaClassifier.map(_.toDebugString).mkString("********Other Tree********\n")
  }

  def numberFeatures():Int={
    val rows=baseTrain.rdd.map(row=>{row.getAs[SparseVector](1).size})
    rows.take(1)(0)
  }

  /**
   * Recursive method for computing feature importances for one tree.
   * This walks down the tree, adding to the importance of 1 feature at each node.
   * @param node  Current node in recursion
   * @param importances  Aggregate feature importances, modified by this method
   */
  private def computeFeatureImportance(
      node: Node,
      importances: OpenHashMap[Int, Double]): Unit = {
    node match {
      case n: InternalNode =>
        val feature = n.split.featureIndex
        val scaledGain = n.gain * n.impurityStats.count
        importances.changeValue(feature, scaledGain, _ + scaledGain)
        computeFeatureImportance(n.leftChild, importances)
        computeFeatureImportance(n.rightChild, importances)
      case n: LeafNode =>
        // do nothing
    }
  }

  /**
   * Normalize the values of this map to sum to 1, in place.
   * If all values are 0, this method does nothing.
   * @param map  Map with non-negative values.
   */
   private def normalizeMapValues(map: OpenHashMap[Int, Double]): Unit = {
    val total = map.map(_._2).sum
    if (total != 0) {
      val keys = map.iterator.map(_._1).toArray
      keys.foreach { key => map.changeValue(key, 0.0, _ / total) }
    }
  }

  }

  def featureImportances(): Vector = {
    val trees: Array[DecisionTreeModel]=metaClassifier.toArray.flatMap(_.trees)
    val numFeatures=numberFeatures
    val totalImportances = new OpenHashMap[Int, Double]()
    trees.foreach { tree =>
      // Aggregate feature importance vector for this tree
      val importances = new OpenHashMap[Int, Double]()
      computeFeatureImportance(tree.rootNode, importances)
      // Normalize importance vector for this tree, and add it to total.
      // TODO: In the future, also support normalizing by tree.rootNode.impurityStats.count?
      val treeNorm = importances.map(_._2).sum
      if (treeNorm != 0) {
        importances.foreach { case (idx, impt) =>
          val normImpt = impt / treeNorm
          totalImportances.update(idx, normImpt, _ + normImpt)
        }
      }
    }
    // Normalize importances
    normalizeMapValues(totalImportances)
    // Construct vector
    val d = if (numFeatures != -1) {
      numFeatures
    } else {
      // Find max feature index used in trees
      val maxFeatureIndex = trees.map(_.maxSplitFeatureIndex()).max
      maxFeatureIndex + 1
    }
    if (d == 0) {
      assert(totalImportances.size == 0, s"Unknown error in computing RandomForest feature" +
        s" importance: No splits in forest, but some non-zero importances.")
    }
    val (indices, values) = totalImportances.iterator.toSeq.sortBy(_._1).unzip
    Vectors.sparse(d, indices.toArray, values.toArray)
  }


  val getNumTrees(): Long={
    numClassifiers
  }





/*
root
 |-- label: double (nullable = true)   SI
 |-- features: vector (nullable = true) SI
 |-- indexedLabel: double (nullable = true)
 |-- indexedFeatures: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)  SI
 |-- prediction: double (nullable = true)
 |-- predictedLabel: string (nullable = true) SI
*/


 }
