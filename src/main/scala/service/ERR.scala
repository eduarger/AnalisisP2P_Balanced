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
 
 def proRatio(values:Vector, inv:Boolean):Double ={
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

 def
 
 
 

}
