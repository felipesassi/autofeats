from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from autofeat.types import Dataset


def make_features_based_on_last_observation_value(df: Dataset) -> DataFrame:
    w = (
        Window()
        .partitionBy(df.public_join_key_col, df.public_join_date_col)
        .orderBy(F.col(df.table_join_date_col).desc())
    )

    table = df.table

    table = table.withColumn("rn", F.row_number().over(w))

    table = table.filter(F.col("rn") == 1).drop("rn")

    return table.select(
        df.public_join_key_col,
        df.public_join_date_col,
        *[F.col(col).alias(f"last___{col}") for col in df.numerical_cols],
    )


def make_features_based_on_first_observation_value(df: Dataset) -> DataFrame:
    w = (
        Window()
        .partitionBy(df.public_join_key_col, df.public_join_date_col)
        .orderBy(df.table_join_date_col)
    )

    table = df.table

    table = table.withColumn("rn", F.row_number().over(w))

    table = table.filter(F.col("rn") == 1).drop("rn")

    return table.select(
        df.public_join_key_col,
        df.public_join_date_col,
        *[F.col(col).alias(f"first___{col}") for col in df.numerical_cols],
    )


def make_features_based_on_rate_between_actual_and_past_value(
    df: Dataset, features: DataFrame, *args, **kwargs
) -> DataFrame:
    w = Window().partitionBy(df.public_join_key_col).orderBy(df.public_join_date_col)

    numerical_cols = features.drop(df.public_join_key_col, df.public_join_date_col).columns

    return features.select(
        df.public_join_key_col,
        df.public_join_date_col,
        *[
            ((F.col(col) - F.lag(col).over(w)) / F.lag(col).over(w)).alias(f"increase_rate_{col}")
            for col in numerical_cols
        ],
    )


def make_features_based_on_lags(df: Dataset, features: DataFrame, *args, **kwargs) -> DataFrame:
    w = Window().partitionBy(df.public_join_key_col).orderBy(df.public_join_date_col)

    n_lags = kwargs.get("options", {"n_lags": [1]})["n_lags"]

    numerical_cols = features.drop(df.public_join_key_col, df.public_join_date_col).columns

    join = lambda x, y: x.join(y, on=[df.public_join_key_col, df.public_join_date_col], how="inner")

    features_list = [
        features.select(
            df.public_join_key_col,
            df.public_join_date_col,
            *[(F.lag(col, n_lag).over(w)).alias(f"lag={n_lag}_{col}") for col in numerical_cols],
        )
        for n_lag in n_lags
    ]

    return reduce(join, features_list)
