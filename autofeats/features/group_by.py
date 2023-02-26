from itertools import combinations
from typing import Any, Callable, Dict, List

import pyspark.sql.functions as F
from pyspark.sql import Column

from autofeats.types import Dataset


def get_categories_from_categorical_data(df: Dataset) -> List[Dict[str, Any]]:
    """
    This function extracts categorical data from the categorical columns. This
    data is transformed in a dictionary with the column name as key and column
    values as values.

    Example of return::

        {"product_type": ["A", "B", "C"]}

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Dict[str, Any]]: List with information about columns categories.
    """
    return [
        {"values": df.or_table.select(c).distinct().toPandas()[c].tolist(), "col_name": c}  # type: ignore
        for c in df.categorical_cols
    ]


def correlation_between_features(df: Dataset) -> List[Column]:
    """
    This function generates the expressions used to calculate the
    correlation between numerical features. All numerical features will be
    combinated into groups of two.

    Example of operation::

        F.corr(F.col("x_1"), F.col("x_2")).alias("corr_between___x_1_x_2")

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Column]: List with the operations to apply in the dataframe
    """
    numerical_cols = df.numerical_cols

    cols_pairs = list(combinations(numerical_cols, 2))

    return [F.corr(c[0], c[1]).alias(f"corr_between___{c[0]}_{c[1]}") for c in cols_pairs]


def numerical_statistics(df: Dataset) -> List[Column]:
    """
    This function generates numerical statisticst about the
    numerical columns in the dataset. The statistics will be
    calculated inside a time window, defined in the dataset
    instantiation.

    Example of operation::

        F.mean(F.col("paid_value")).alias("mean___paid_value")

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Column]: List with the operations to apply in the dataframe
    """

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
    """
    This function counts the occorurences of each category inside a
    categorical column.

    Example of operation::

        F.count(F.when(F.col("product_type") == "A", F.col("product_type")))

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Column]: List with the operations to apply in the dataframe
    """

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
    """
    This function will apply count and countDistinct to
    the categorical columns as a whole.

    Example of operation::

        F.count(F.col("product_type"))

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Column]: List with the operations to apply in the dataframe
    """
    functions: List[Callable[[Column], Column]] = [F.count, F.countDistinct]

    categories_to_analyze = get_categories_from_categorical_data(df)

    return [
        function(F.col(categorie["col_name"])).alias(
            f"{function.__name__}___{categorie['col_name']}"
        )
        for categorie in categories_to_analyze
        for function in functions
    ]


def categorical_statistics(df: Dataset) -> List[Column]:
    """
    This function applies count_occurences_of_each_category
    and count_categorical_values to the dataset.

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Column]: List with the operations to apply in the dataframe
    """
    return count_occurences_of_each_category(df) + count_categorical_values(df)


def statistics_of_numerical_data_in_categorical_groups(
    df: Dataset,
) -> List[Column]:
    """
    This function will calculate numerical statistics using a
    pivoted version of the dataset.

    Example of operation::

        F.mean(F.when(F.col("product_type") == "A", F.col("paid_value")))

    Args:
        df (Dataset): Dataset initialized with necessary information

    Returns:
        List[Column]: List with the operations to apply in the dataframe
    """
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
