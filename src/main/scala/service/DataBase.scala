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
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint


class DataBase(tableBase: String,  numP:Int, sqlContext: SQLContext, filters: String) extends Serializable {

  private val cols="""idn,resp_code,label,fraude,nolabel,monto,valida_cifin,confiabilidad_documento,first_tarjeta,
  first_email,first_doc,coc_act_max_tarjeta,coc_act_pro_tarjeta,coc_act_max_email,
  coc_act_pro_email,coc_act_max_doc,coc_act_pro_doc,COP,OTH,USD,EUR,fds,morn,tard,
  noch,madr,probabilidad_franjaindex_tarjetac,probabilidad_franjaindex_documentoclientec,
  probabilidad_franjaindex_email,probabilidad_ubicacion_tarjetac,probabilidad_ubicacion_documentoclientec,
  probabilidad_ubicacion_email,probabilidad_id_comercio_tarjetac,probabilidad_id_comercio_documentoclientec,
  probabilidad_id_comercio_email,probabilidad_id_sector_tarjetac,probabilidad_id_sector_documentoclientec,
  probabilidad_id_sector_email,probabilidad_ip_tarjetac,probabilidad_ip_documentoclientec,probabilidad_ip_email,
  probabilidad_retail_code_tarjetac,probabilidad_retail_code_documentoclientec,probabilidad_retail_code_email,
  probabilidad_nav_tarjetac,probabilidad_nav_documentoclientec,probabilidad_nav_email,probabilidad_os_tarjetac,
  probabilidad_os_documentoclientec,probabilidad_os_email,coc_mon_cuo,coc_cuotas_cuotaspro,emailnull,
  ubicacionull,tarjetanull,sectionnull,comercionull,retailnull,documentoclientecnull,
  ipnull,punto_de_venta,nat_flag,cuenta_monto_tarjetac_6h,pro_monto_tarjetac_6h,cuenta_monto_tarjetac_24h,
  pro_monto_tarjetac_24h,cuenta_monto_tarjetac_15d,pro_monto_tarjetac_15d,cuenta_monto_tarjetac_30d,
  pro_monto_tarjetac_30d,cuenta_monto_tarjetac_60d,pro_monto_tarjetac_60d,cuenta_monto_tarjetac_365d,
  pro_monto_tarjetac_365d,cuenta_monto_documentoclientec_6h,pro_monto_documentoclientec_6h,
  cuenta_monto_documentoclientec_24h,pro_monto_documentoclientec_24h,cuenta_monto_documentoclientec_15d,
  pro_monto_documentoclientec_15d,cuenta_monto_documentoclientec_30d,pro_monto_documentoclientec_30d,
  cuenta_monto_documentoclientec_60d,pro_monto_documentoclientec_60d,cuenta_monto_documentoclientec_365d,
  pro_monto_documentoclientec_365d,cuenta_monto_email_6h,pro_monto_email_6h,cuenta_monto_email_24h,
  pro_monto_email_24h,cuenta_monto_email_15d,pro_monto_email_15d,cuenta_monto_email_30d,pro_monto_email_30d,
  cuenta_monto_email_60d,pro_monto_email_60d,cuenta_monto_email_365d,pro_monto_email_365d,
  cuenta_retail_code_tarjetac_30d,cuenta_retail_code_documentoclientec_30d,cuenta_retail_code_email_30d,
  cuenta_monto_ip_6h,cuenta_monto_ip_24h,cuenta_monto_ip_10m,cuenta_monto_cp40_bine_6h,
  cuenta_monto_cp40_bine_24h,cuenta_monto_cp40_bine_15d,cuenta_monto_cp40_bine_30d,cuenta_monto_cp40_bine_60d,
  cuenta_monto_cp40_bine_365d,sum_apr_retail_code_15d,count_apr_retail_code_30d,sum_apr_retail_code_30d,
  count_apr_retail_code_60d,sum_apr_retail_code_60d,coc_6h_can_tx_rec_apr_retail,
  coc_24h_can_tx_rec_apr_retail,coc_30d_can_tx_rec_apr_retail,coc_60d_can_tx_rec_apr_retail,
  coc_6h_sum_tx_rec_apr_retail,coc_24h_sum_tx_rec_apr_retail,coc_30d_sum_tx_rec_apr_retail,
  coc_60d_sum_tx_rec_apr_retail,coc_6h_pro_nat_retail_code,coc_24h_pro_nat_retail_code,
  coc_15d_pro_nat_retail_code,coc_30d_pro_nat_retail_code,coc_60d_pro_nat_retail_code,
  coc_6h_pro_int_retail_code,coc_24h_pro_int_retail_code,coc_15d_pro_int_retail_code,
  coc_30d_pro_int_retail_code,coc_60d_pro_int_retail_code,probabilidad_nat_flag_retail_code_6h,
  probabilidad_nat_flag_retail_code_24h,probabilidad_nat_flag_retail_code_15d,
  probabilidad_nat_flag_retail_code_30d,probabilidad_nat_flag_retail_code_60d,max_match,
  match_tarjeta_documento_cliente,cant_same_data,navnull,osnull"""

  val query="SELECT " + cols+ " FROM "+ tableBase
  private val dataFrameBase={
    if ("".equalsIgnoreCase(filters))
      sqlContext.sql(query).coalesce(numP)
    else
      sqlContext.sql(query).where(filters).coalesce(numP)
    }
  private val logger = LogManager.getLogger("DataBase")


  def getDataFrame():DataFrame={
    val retorno=dataFrameBase
    retorno
  }

  def getDataFrameNum(preCal:Boolean):DataFrame={
    val tName=tableBase+"_num"
    val retorno={
      if(preCal)
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)
      else
        dataFrameBase.drop("resp_code").drop("fraude").drop("nolabel").write.mode(SaveMode.Overwrite).saveAsTable(tName)
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)
    }
    retorno
  }

  def getDataFrameLabeledLegalFraud(preCal:Boolean):DataFrame={
    val tName=tableBase+"_labeled"
    val base=dataFrameBase.where("label!=0")
    val names = base.columns
    val lon=names.length
    // se definen os atributos que no son variables
    val ignore = Array("idn", "label","resp_code","fraude","nolabel")
    val assembler = (new VectorAssembler()
    .setInputCols( for (i <- names if !(ignore contains i )) yield i)
    .setOutputCol("features"))
    val retorno={
      if(preCal){
        logger.info("........Reading  "+tName +"..............")
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)

      } else{
        logger.info("........Converting to features...............")
        val data = assembler.transform(base)
        logger.info("..........Conviertinedo DF a labeling...............")
        val rows: RDD[Row] = data.rdd
        val labeledPoints: RDD[LabeledPoint]=(rows.map(row =>{LabeledPoint(row.getInt(2).toDouble,
        row.getAs[SparseVector](lon))}))
        import sqlContext.implicits._
        val labeledDF=labeledPoints.toDF()
        logger.info("........writing  "+tName +"..............")
        labeledDF.write.mode(SaveMode.Overwrite).saveAsTable(tName)
        logger.info("........Reading  "+tName +"..............")
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)
   }
  }
  retorno
}

}
