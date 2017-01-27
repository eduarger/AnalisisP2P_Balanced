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
import org.apache.spark.sql.expressions.Window

class EER()  extends Serializable {

 @transient val logger = LogManager.getLogger("ERR Computation")


  val proRatio = (values:Vector,inv:Boolean) => {
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

  val sqlProRatio = udf(proRatio)

  def returnOneOrZero = udf[Int,Boolean](arg => if (arg) 1 else 0)

  def getLikeHoodRatio(scores:DataFrame, name:String, inv :Boolean):DataFrame={
   val res=scores.select(name).withColumn("lr", sqlProRatio(col(name),lit(inv)))
   res
 }


  def compute(trueSet:DataFrame,falseSet:DataFrame, tol:Double):Vector={
    logger.info("..........procesing scores...............")
    val trueScores=(getLikeHoodRatio(trueSet,"probability",true)
      .withColumn("flag",returnOneOrZero(lit(true))))
    val falseScores=(getLikeHoodRatio(falseSet,"probability",true)
      .withColumn("flag",returnOneOrZero(lit(false))))
    val iNumTrue=trueScores.count
    val iNumFalse=falseScores.count
    val iTotal=iNumTrue+iNumFalse
    val w = Window.orderBy(col("lr")).rowsBetween(Long.MinValue,0)
    val scores = (trueScores.unionAll(falseScores)
    .withColumn("vSumFalse",sum(col("flag")).over(w))
    .withColumn("aux", monotonicallyIncreasingId)
    .withColumn("vSumTrue", col("aux")-col("vSumFalse"))
    .withColumn("vSumFalse2", lit(iNumFalse)-col("vSumFalse"))
    .withColumn("Pmiss", col("vSumTrue")/lit(iNumTrue))
    .withColumn("Pfa", col("vSumFalse2")/lit(iNumFalse))
    .withColumn("mResta", abs(col("Pmiss")-col("Pfa")))
    )
    scores.cache()
    logger.info("..........getting counts of score...............")
    val minR=scores.select(min(col("mResta"))).collect
    val minResta=minR(0).getDouble(0)
    val EERdf=scores.where("mResta="+minResta)
    val numZeros=EERdf.count()
    val eer={
      if(minResta>tol && numZeros>=1){
        logger.info("WARN:the EER does not meet with the tolerance!! the returned values are the minimun")
        val range=EERdf.select("Pfa", "lr").collect
        val i=math.round(range.length/2.0).toInt
        Vectors.dense(range(i-1).getDouble(0)*100,range(i-1).getDouble(1), 0)

      }
      else if(minResta<=tol && numZeros>=1){
        val range=EERdf.select("Pfa", "lr").collect
        val i=math.round(range.length/2.0).toInt
        Vectors.dense(range(i-1).getDouble(0)*100,range(i-1).getDouble(1), 1)
      }
      else{
        logger.info("WARN:No EER computed due some error!")
        Vectors.dense(0.0,0.0, 0)
      }
      }
      EERdf.unpersist()
      eer
    }

    def computePlots(trueSet: DataFrame, falseSet: DataFrame, params: (String, Int, Int), k:Int, bins:Int, clase: String):String={
      val trueScores=getLikeHoodRatio(trueSet,"probability",true)
      val falseScores=getLikeHoodRatio(falseSet,"probability",true)
      val limits=trueScores.unionAll(falseScores).select(min(col("lr")), max(col("lr"))).collect
      val xmin=limits(0).getDouble(0)
      val xmax=limits(0).getDouble(1)
      val increment = (xmax - xmin) / (bins - 1)
      val x = List.tabulate(bins)(i => xmin +  increment * i).toArray
      val trueNum = trueScores.select("lr").map { _.getDouble(0)}.histogram(x)
      val cumfa=trueNum.scanLeft(0)(_.toInt+_.toInt)
      val numfa=cumfa.map{_.toDouble/cumfa.max}
      val falseNum = falseScores.select("lr").map { _.getDouble(0)}.histogram(x)
      val cumfr=falseNum.scanLeft(0)(_.toInt+_.toInt)
      val numfr=cumfr.map{_.toDouble/cumfr.max}
      val histnumFa=for ((value, index) <- x.zipWithIndex)
        yield  (value, numfa(index))
      val histnumFr=for ((value, index) <- x.zipWithIndex)
        yield  (value, numfr(index))
      var out=histnumFa.mkString(","+k+","+clase+"-fraude,"+params + "\n")+","+k+","+"fraude,"+params+"\n" filterNot ("()" contains _)
      out=out+histnumFr.mkString(","+k+","+clase+"-legal,"+params + "\n")+","+k+","+"legal,"+params+"\n" filterNot ("()" contains _)
      out
    }

}
