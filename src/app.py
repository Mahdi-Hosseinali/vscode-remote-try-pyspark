# Sample Code - just press F5

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([("10", "Bob"), ("11", "JimBob"), ("13", "Bobby")], ["age", "Name"])

df.show(20, False)