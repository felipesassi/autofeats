from itertools import combinations
from typing import Any, Callable, Dict, List

import pyspark.sql.functions as F
from pyspark.sql import Column

from autofeat.types import Dataset


def get_categories_from_categorical_data(df: Dataset) -> List[Dict[str, Any]]:
    return [
        {"values": df.table.select(c).distinct().toPandas()[c].tolist(), "col_name": c}  # type: ignore
        for c in df.categorical_cols
    ]


def make_features_based_on_correlation_between_features(df: Dataset) -> List[Column]:
    numerical_cols = df.numerical_cols

    cols_pairs = list(combinations(numerical_cols, 2))

    return [F.corr(c[0], c[1]).alias(f"corr_between___{c[0]}_{c[1]}") for c in cols_pairs]


def make_features_based_on_numerical_statistics(df: Dataset) -> List[Column]:
    functions: List[Callable[[Column], Column]] = [
        F.sum,
        F.mean,
        F.stddev,
        F.min,
        F.max,
        F.kurtosis,
        F.skewness,
    ]

    numerical_cols = df.numerical_cols

    return [
        function(numerical_col).alias(f"{function.__name__}___{numerical_col}")
        for function in functions
        for numerical_col in numerical_cols
    ]


def count_occurences_of_each_category(df: Dataset) -> List[Column]:
    functions: List[Callable[[Column], Column]] = [F.count]

    categories_to_analyze = get_categories_from_categorical_data(df)

    return [
        function(F.when(F.col(categorie["col_name"]) == c, F.col(categorie["col_name"]))).alias(
            f"{categorie['col_name']}={c}__{function.__name__}___{categorie['col_name']}"
        )
        for categorie in categories_to_analyze
        for c in categorie["values"]
        for function in functions
    ]


def count_categorical_values(df: Dataset) -> List[Column]:
    functions: List[Callable[[Column], Column]] = [F.count, F.countDistinct]

    categories_to_analyze = get_categories_from_categorical_data(df)

    return [
        function(F.col(categorie["col_name"])).alias(
            f"{function.__name__}___{categorie['col_name']}"
        )
        for categorie in categories_to_analyze
        for function in functions
    ]


def make_features_based_on_categorical_statistics(df: Dataset) -> List[Column]:
    return count_occurences_of_each_category(df) + count_categorical_values(df)


def make_features_based_on_statistics_of_numerical_data_in_categorical_groups(
    df: Dataset,
) -> List[Column]:
    functions: List[Callable[[Column], Column]] = [
        F.sum,
        F.mean,
        F.stddev,
        F.min,
        F.max,
        F.kurtosis,
        F.skewness,
    ]

    numerical_cols = df.numerical_cols

    categories_to_analyze = get_categories_from_categorical_data(df)

    return [
        function(F.when(F.col(categorie["col_name"]) == c, F.col(numerical_col))).alias(
            f"{categorie['col_name']}={c}__{function.__name__}___{numerical_col}"
        )
        for categorie in categories_to_analyze
        for c in categorie["values"]
        for function in functions
        for numerical_col in numerical_cols
    ]
