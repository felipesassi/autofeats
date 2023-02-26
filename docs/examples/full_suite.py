import pyspark.sql.functions as F
from examples.data import make_transactions
from pyspark.sql import DataFrame, SparkSession

from autofeats import make_features
from autofeats.types import Dataset

spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.executor.memory", "6g")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

transactions = spark.createDataFrame(make_transactions())

public = transactions.groupby(F.col("consumer_id").alias("consumer_id_ref")).agg(
    F.max("paymnent_date").alias("date_ref")
)

df = Dataset(
    table=transactions,
    primary_key_col="transaction_id",
    table_join_key_col="consumer_id",
    table_join_date_col="paymnent_date",
    numerical_cols=["paid_value", "discount"],
    categorical_cols=["product_type", "buy_type"],
    public=public,
    public_join_key_col="consumer_id_ref",
    public_join_date_col="date_ref",
)

features: DataFrame = make_features.run(
    df=df,
    suites=[
        "numerical_statistics",
        "numerical_in_categorical_groups",
        "correlation",
        "categorical_statistics",
        "first_observation_features",
        "last_observation_features",
        "lags",
        "increase_rate",
    ],
    options={"n_lags": [1]},
)

print(f"Features created: {features.columns}")

print(f"Number of features: {len(features.columns)}")
