from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from autofeats.types import Dataset


def last_observation_value(df: Dataset) -> DataFrame:
    """
    This function will return the last row (observation) in
    the dataset. The columns selected will be only the numerical
    ones.

    Example::

        w = Window().partitionBy("customer_id").orderBy(F.col("buy_date").desc())

        df = df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        DataFrame: Dataframe with the features
    """
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


def first_observation_value(df: Dataset) -> DataFrame:
    """
    This function will return the first row (observation) in
    the dataset. The columns selected will be only the numerical
    ones.

    Example::

        w = Window().partitionBy("customer_id").orderBy("buy_date")

        df = df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")


    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        DataFrame: Dataframe with the features
    """
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


def rate_between_actual_and_past_value(
    df: Dataset, features: DataFrame, *args, **kwargs
) -> DataFrame:
    """
    This function should be applied to the generated features.
    This function generates the increase rate comparing the feature value
    in a date X with the feature value in a date X - 1.

    Example::

        w = Window().partitionBy("customer_id").orderBy("date_ref")
        df = df.withColumn("increase_rate_mean___paid_value, (F.lag("paid_value").over(w) - F.col("paid_value"))/F.col("paid_value"))

    Args:
        df (Dataset): Dataset initialized with necessary information
        features (DataFrame): Dataframe with features

    Returns:
        DataFrame: Dataframe with the features
    """
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


def lags(df: Dataset, features: DataFrame, *args, **kwargs) -> DataFrame:
    """
    This function should be applied to the generated features.
    This function applies a lag function to the features table.

    Example::

        w = Window().partitionBy("customer_id").orderBy("date_ref")
        df = df.withColumn("lag=1_mean___paid_value", F.lag("mean___paid_value").over(w))

    Args:
        df (Dataset): Dataset initialized with necessary information
        features (DataFrame): Dataframe with features

    Returns:
        DataFrame: Dataframe with the features
    """
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
