name := "AnalisisP2P_Balanced"
version := "2.0"
scalaVersion := "2.11.8"
resolvers += Resolver.sonatypeRepo("public")
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.1.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.1.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.1.0" % "provided",
  "org.apache.spark" %% "spark-hive" % "2.1.0" % "provided",
  "com.github.scopt" %% "scopt" % "3.4.0"
)
