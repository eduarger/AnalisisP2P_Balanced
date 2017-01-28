package config
import java.io.File
case class Config(in:String="base_test",par: Int = 250, read: String = "",
  out: String = "sal", kfolds: Int = 5, imp: Seq[String] = Seq("entropy"),
  depth: Seq[Int] = Seq(10), bins: Seq[Int] = Seq(64),
  axes: Seq[Int]=Seq(-10,10),train: Double=0.7,
  filter: String ="", densidad: Boolean=false)
