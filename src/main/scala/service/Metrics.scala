package service
import org.apache.spark.sql.DataFrame
import org.apache.log4j.LogManager
import org.apache.spark.SparkContext
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{types, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{SparseVector, DenseVector,Vectors,Vector}


class Metrics(dataSet: DataFrame, labels:Array[Double]) extends Serializable {
  val labelPos=labels(0)
  val labelNeg=labels(1)
  val predRow: RDD[Row]=dataSet.select("label", "predictedLabel").rdd
  val predRDD: RDD[(Double, Double)] = (predRow.map(row=>{(row.getDouble(0), row.getDouble(1).toDouble)}))
  val tp=predRDD.filter(r=>r._1== labelPos && r._2==labelPos).count().toDouble
  val fn=predRDD.filter(r=>r._1== labelPos && r._2== labelNeg).count().toDouble
  val tn=predRDD.filter(r=>r._1== labelNeg && r._2== labelNeg).count().toDouble
  val fp=predRDD.filter(r=>r._1== labelNeg && r._2== labelPos).count().toDouble
  val sens = (tp/(tp+fn))*100.0
  val spc = (tn/(fp+tn))*100.0
  val pre= (tp/(tp+fp))*100.0
  val acc= ((tp+tn)/(tp+fn+fp+tn))*100.0
  val f1= ((2*pre*sens)/(pre+sens))
  val mGeo=math.sqrt(sens*spc)
  val pExc=(tp*tn-fn*fp)/((fn+tp)*(tn+fp))
  val MCC=(tp*tn-fp*fn)/math.sqrt((fn+tp)*(tn+fp)*(fp+tp)*(fn+tn))

}
