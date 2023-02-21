import pyspark.sql.functions as F

from autofeat import __version__
from autofeat.types import Dataset
from tests.utils import make_transactions, spark


def test_dataset(spark):
    df = spark.createDataFrame(make_transactions())

    public = df.groupby(F.col("consumer_id").alias("consumer_id_ref")).agg(
        F.max("paymnent_date").alias("date_ref")
    )

    df = Dataset(
        table=df,
        primary_key_col="transaction_id",
        table_join_key_col="consumer_id",
        table_join_date_col="paymnent_date",
        numerical_cols=["paid_value", "discount"],
        categorical_cols=["product_type"],
        public=public,
        public_join_key_col="consumer_id_ref",
        public_join_date_col="date_ref",
    )

    assert df.table.columns == [
        "consumer_id_ref",
        "date_ref",
        "transaction_id",
        "consumer_id",
        "paymnent_date",
        "paid_value",
        "discount",
        "product_type",
    ]
