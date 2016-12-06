name := "AnalisisP2P_Balanced"
version := "1.0"
scalaVersion := "2.10.5"
resolvers += Resolver.sonatypeRepo("public")
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-hive" % "1.6.0" % "provided",
  "com.timesprint" %% "hasher" % "0.3.0",
  "com.github.nscala-time" %% "nscala-time" % "2.12.0",
  "com.databricks" %% "spark-csv" % "1.4.0",
  "mysql" % "mysql-connector-java" % "5.1.16",
  "com.github.scopt" %% "scopt" % "3.4.0"
)