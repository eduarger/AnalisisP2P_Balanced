package service
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import scala.collection.mutable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors,Vector}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{types, _}
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.tree.model.Node
import scala.collection.mutable.OpenHashMap // instead of private class import org.apache.spark.util.collection.OpenHashMap
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
    sqlContext: SQLContext
    )
    extends Serializable {

   val numPartitions=numP
   @transient private val logger = LogManager.getLogger("RandomForestWithBalance")
   private var metaClassifier = scala.collection.mutable.ArrayBuffer.empty[RandomForestModel]//[PipelineModel]
   // training dataset empty, in this dataset will be saved the balanced dataset
   @transient val labelMinor=labels(0)
   @transient val labelMayor=labels(1)
   // relation for balance the training datasets
   val numOfMinorClass= baseTrain.where("label="+labelMinor).count()
   val numOfMayorClass= baseTrain.where("label!="+labelMinor).count()
   logger.info("Minor class " +numOfMinorClass)
   logger.info("Mayor class " +numOfMayorClass)
   // calculate the smaple fraction and the number of classifiers
   private val sampleFraction=numOfMinorClass.toDouble/numOfMayorClass.toDouble
   private val fractionInv=numOfMayorClass/numOfMinorClass
   // avoid odd values
   private val  numClassifiers=if (fractionInv%2==0) fractionInv + 1 else fractionInv
   // indexers and converters
   private val featureIndex=feIndx
   val categoricalFeaturesInfo=feIndx.categoryMaps
   val cat=categoricalFeaturesInfo.map(kv => (kv._1, kv._2.size))
   //labeles
   logger.info("Number of classifiers(trees): "+numClassifiers)
   logger.info("sample rate is: "+sampleFraction)

  //set the training dataset
  @transient def getTrainingBalancedSet() : DataFrame ={
    val dfMinor= baseTrain.where("label="+labelMinor)
    val dfMayor= baseTrain.where("label!="+labelMinor).sample(false,sampleFraction)
    val res=dfMinor.unionAll(dfMayor)
    res
   }
  //Convert dataframe to RDD with labeled parsed
  def toRDDLabeledParsed(dataInput:DataFrame):RDD[LabeledPoint]={
    val rows=dataInput.rdd
    @transient val lm=labelMinor
    val labeledData: RDD[LabeledPoint]=rows.map(row =>{
      // convert the labels if is equal to labelMinor so 1.0 for other 0.0
      val newLabel= if(row.getDouble(0)==lm) 1.0 else 0.0
      val res=LabeledPoint(newLabel,row.getAs[SparseVector](1))
      res})
    labeledData
  }

  //training one tree
  private def trainingTree() : RandomForestModel  = {
    // set the training dataset in each traiing
    @transient val data=getTrainingBalancedSet()
    @transient
    val labeledPoints: RDD[LabeledPoint]=toRDDLabeledParsed(data)
    val model = RandomForest.trainClassifier(labeledPoints, 2, cat,
    1, "auto", parameters._1, parameters._2, parameters._3)
    model
    }

  // training all
  @transient def training() : Unit  = {
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

  // get prediction and label from nodes of the trees
  def predictNodeProb(nodo: Node,features: Vector) : Array[Double] = {
    if (nodo.isLeaf) {
      Array(nodo.predict.predict,nodo.predict.prob)
    } else {
      if (nodo.split.get.featureType == Continuous) {
        if (features(nodo.split.get.feature) <= nodo.split.get.threshold) {
          predictNodeProb(nodo.leftNode.get, features)
        } else {
          predictNodeProb(nodo.rightNode.get, features)
        }
      } else {
        if (nodo.split.get.categories.contains(features(nodo.split.get.feature))) {
          predictNodeProb(nodo.leftNode.get, features)
        } else {
          predictNodeProb(nodo.rightNode.get, features)
        }
      }
    }
  }

// function to get predictions and probilities of the trees
  def predictWithProb(tree: DecisionTreeModel,features: Vector) : Array[Double] = {
    predictNodeProb(tree.topNode, features)
    }
// average a list taht reprents the output of set o decision trees
  def avgList(lista: ArrayBuffer[Array[Double]]): Double={
    lista.map(_(1)).sum/lista.size
  }

// get the predictions in soft way!
  def labelAndProbSoft(results: ArrayBuffer[Array[Double]]) : (Double,Double,Double)={
    val predOnes=results.filter(_(0)==1.0).union(results.filterNot(_(0)==1.0).map(a=>Array(1.0,1.0-a(1))))
    val countOne=results.filter(_(0)==1.0).size
    val countZero=results.filterNot(_(0)==1.0).size
    val predZeros=results.filterNot(_(0)==1.0).union(results.filter(_(0)==1.0).map(a=>Array(0.0,1.0-a(1))))
    val proClaseUno=avgList(predOnes)
    val proClaseCero=avgList(predZeros)
    val predLabel=
      if(proClaseUno>proClaseCero || countOne>countZero)
        labelMinor
      else if(proClaseUno<proClaseCero || countOne<countZero)
        labelMayor
      else
        labelMayor
    (predLabel,proClaseCero,proClaseUno)
  }

  // predcit one sample of a RDD depending of the Startegy
  def predictOneSampleProb(sample: LabeledPoint,
      models: scala.collection.mutable.ArrayBuffer[RandomForestModel],
      estrategia: String="soft") : (Double,Vector,Vector,Double)= {
    val resultado:  (Double,Vector,Vector,Double)={
    val mapResults=metaClassifier.map(mod => predictWithProb(mod.trees(0),sample.features))
      if(estrategia.equalsIgnoreCase("soft")) {
        //chek the probilities of each class
        val predAndProb=labelAndProbSoft(mapResults)
        val originalLabel=if(sample.label==1) labelMinor else labelMayor
        (originalLabel,sample.features,Vectors.dense(predAndProb._2,predAndProb._3), predAndProb._1)
    } else if(estrategia.equalsIgnoreCase("hard")){
        val proClaseUno=mapResults.map(_(0)).filter(_==1).size.toDouble/models.size
        val proClaseCero=mapResults.map(_(0)).filterNot(_==1).size.toDouble/models.size
        val predLabel=if(proClaseUno>proClaseCero) labelMinor else labelMayor
        val originalLabel=if(sample.label==1) labelMinor else labelMayor
        (originalLabel,sample.features,Vectors.dense(Array(proClaseCero,proClaseUno)), predLabel)
    } else {
        println("Value strategy Not Defined!!!!!!!!!!! using soft as default")
        predictOneSampleProb(sample,models, "soft")
    }
      }
    resultado
  }
  // predict the result of one dataframe
  def getPredictions(testData:DataFrame,
      sc:SparkContext,
      estrategia: String="soft") : DataFrame ={
    @transient val labeledPoints: RDD[LabeledPoint]=toRDDLabeledParsed(testData)
    //broadcast all the trees trained
    @transient val arboles=metaClassifier
    @transient val labelsCopy=labels
    @transient val treesBroadcasted=sc.broadcast(arboles)
    val rddPredictions=labeledPoints.map(point=>{
    predictOneSampleProb(point,treesBroadcasted.value,estrategia)})
    //convert to dataframe
    import sqlContext.implicits._
    val dataFramePredcited=rddPredictions.toDF("label", "features","probability","predictedLabel")
    dataFramePredcited
  }
  /**
   * Method to export to a comprenhisive text all the trees trained
   */
  def toDebugString():String ={
    val texto=metaClassifier.map(_.toDebugString).mkString("********Other Tree********\n")
    texto
  }

  def numberFeatures():Int={
    val rows=baseTrain.rdd.map(row=>{row.getAs[SparseVector](1).size})
    rows.take(1)(0)
  }

  private def computeFeatureImportance(
      node: Node,
      importances: OpenHashMap[Int, Double]): Unit = {
    val res={
      if(!node.isLeaf){
        val feature = node.split.get.feature
        val scaledGain = node.stats.get.gain
        importances.put(feature, importances.get(feature).getOrElse(0.0) + scaledGain)
        computeFeatureImportance(node.leftNode.get, importances)
        computeFeatureImportance(node.rightNode.get, importances)
      }
    }

    }


   private def normalizeMapValues(map: OpenHashMap[Int, Double]): Unit = {
    val total = map.map(_._2).sum
    if (total != 0) {
      val keys = map.iterator.map(_._1).toArray
      keys.foreach { key => map.put(key, map.get(key).getOrElse(0.0)/total) }

    }
  }

  def maxSplitFeatureIndex(node: Node): Int = {
      math.max(node.split.get.feature,
        math.max(maxSplitFeatureIndex(node.leftNode.get), maxSplitFeatureIndex(node.rightNode.get)))
    }

  def featureImportances(): Vector = {
    val trees: Array[DecisionTreeModel]=metaClassifier.toArray.flatMap(_.trees)
    val numFeatures=numberFeatures
    val totalImportances = new OpenHashMap[Int, Double]()
    trees.foreach { tree =>
      // Aggregate feature importance vector for this tree
      val importances = new OpenHashMap[Int, Double]()
      computeFeatureImportance(tree.topNode, importances)
      // Normalize importance vector for this tree, and add it to total.
      // TODO: In the future, also support normalizing by tree.rootNode.impurityStats.count?
      val treeNorm = importances.map(_._2).sum
      if (treeNorm != 0) {
        importances.foreach { case (idx, impt) =>
          val normImpt = impt / treeNorm
          totalImportances.put(idx, totalImportances.get(idx).getOrElse(0.0) + normImpt)
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
      val maxFeatureIndex = trees.map(t=>maxSplitFeatureIndex(t.topNode)).max
      maxFeatureIndex + 1
    }
    if (d == 0) {
      assert(totalImportances.size == 0, s"Unknown error in computing RandomForest feature" +
        s" importance: No splits in forest, but some non-zero importances.")
    }
    val (indices, values) = totalImportances.iterator.toSeq.sortBy(_._1).unzip
    Vectors.sparse(d, indices.toArray, values.toArray)
  }

  def getNumTrees(): Long={
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
