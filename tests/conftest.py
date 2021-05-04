

from pyspark.sql import SparkSession
import pytest

@pytest.fixture(scope='session')
def get_spark_session():
    spark = SparkSession.builder.getOrCreate()
    return spark