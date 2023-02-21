import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from autofeats.types import Dataset


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def make_transactions():
    return pd.DataFrame(
        data={
            "transaction_id": range(12),
            "consumer_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "paid_value": [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
            "discount": [2, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34],
            "product_type": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "A", "A", "A"],
            "paymnent_date": [
                "2022-01-01",
                "2022-01-01",
                "2022-01-01",
                "2022-01-02",
                "2022-01-02",
                "2022-01-02",
                "2022-01-03",
                "2022-01-03",
                "2022-01-03",
                "2022-01-04",
                "2022-01-04",
                "2022-01-04",
            ],
        }
    )


@pytest.fixture(scope="session")
def make_dataset(spark):
    transactions = spark.createDataFrame(make_transactions())

    public = transactions.groupby(F.col("consumer_id").alias("consumer_id_ref")).agg(
        F.max("paymnent_date").alias("date_ref")
    )

    return Dataset(
        table=transactions,
        primary_key_col="transaction_id",
        table_join_key_col="consumer_id",
        table_join_date_col="paymnent_date",
        numerical_cols=["paid_value"],
        categorical_cols=["product_type"],
        public=public,
        public_join_key_col="consumer_id_ref",
        public_join_date_col="date_ref",
    )


@pytest.fixture(scope="session")
def features(spark):
    return spark.createDataFrame(
        pd.DataFrame(
            data={
                "consumer_id_ref": [1, 1, 1],
                "transaction_id": [1, 2, 3],
                "date_ref": ["2022-01-04", "2022-01-05", "2022-01-06"],
                "first___paid_value": [1, 3, 6],
            }
        )
    ).drop("transaction_id")


@pytest.fixture(scope="session")
def make_dataset_to_correlation(spark):
    transactions = spark.createDataFrame(make_transactions())

    public = transactions.groupby(F.col("consumer_id").alias("consumer_id_ref")).agg(
        F.max("paymnent_date").alias("date_ref")
    )

    return Dataset(
        table=transactions,
        primary_key_col="transaction_id",
        table_join_key_col="consumer_id",
        table_join_date_col="paymnent_date",
        numerical_cols=["paid_value", "discount"],
        categorical_cols=["product_type"],
        public=public,
        public_join_key_col="consumer_id_ref",
        public_join_date_col="date_ref",
    )
