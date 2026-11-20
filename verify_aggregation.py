import sys
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("verify").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sample = spark.read.parquet("output/labelled.parquet").limit(100000).cache()
spark_counts = {r["label"]: r["n"] for r in sample.groupBy("label").count().withColumnRenamed("count", "n").collect()}
pdf = sample.select("label").toPandas()
pandas_counts = pdf["label"].value_counts().to_dict()
print("Spark :", spark_counts)
print("pandas:", pandas_counts)
print("MATCH" if spark_counts == pandas_counts else "MISMATCH")
spark.stop()
