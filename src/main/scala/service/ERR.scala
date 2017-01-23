package service
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors,Vector}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{types, _}

class ERR(trueScores:DataFrame, falseScores: DataFrame )  extends Serializable {

 @transient val logger = LogManager.getLogger("ERR Computation")

}
