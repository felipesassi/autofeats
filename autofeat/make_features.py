from functools import reduce
from typing import Optional

from pyspark.sql import DataFrame

from autofeat.features.group_by import (
    make_features_based_on_categorical_statistics,
    make_features_based_on_correlation_between_features,
    make_features_based_on_numerical_statistics,
    make_features_based_on_statistics_of_numerical_data_in_categorical_groups,
)
from autofeat.features.window import (
    make_features_based_on_first_observation_value,
    make_features_based_on_lags,
    make_features_based_on_last_observation_value,
    make_features_based_on_rate_between_actual_and_past_value,
)
from autofeat.types import Dataset

OPERATIONS = {
    "numerical_statistics": make_features_based_on_numerical_statistics,
    "numerical_in_categorical_groups": make_features_based_on_statistics_of_numerical_data_in_categorical_groups,
    "correlation": make_features_based_on_correlation_between_features,
    "categorical_statistics": make_features_based_on_categorical_statistics,
}

FIRST_LAST = {
    "first_observation_features": make_features_based_on_first_observation_value,
    "last_observation_features": make_features_based_on_last_observation_value,
}

LAG = {
    "lags": make_features_based_on_lags,
    "increase_rate": make_features_based_on_rate_between_actual_and_past_value,
}


def run(df: Dataset, suites: list, options: dict) -> Optional[DataFrame]:
    features = None

    join = lambda x, y: x.join(y, on=[df.public_join_key_col, df.public_join_date_col], how="inner")

    if (
        ("numerical_statistics" in suites)
        or ("numerical_in_categorical_groups" in suites)
        or ("correlation" in suites)
        or ("categorical_statistics" in suites)
    ):
        aggregations = reduce(
            lambda x, y: x + y,
            [OPERATIONS[suite](df=df) for suite in suites if suite in OPERATIONS],
        )

        features = df.table.groupby(df.public_join_key_col, df.public_join_date_col).agg(
            *aggregations
        )

    if ("first_observation_features" in suites) or ("last_observation_features" in suites):
        if features is not None:
            features = features.join(
                reduce(join, [FIRST_LAST[suite](df) for suite in suites if suite in FIRST_LAST]),
                on=[df.public_join_key_col, df.public_join_date_col],
                how="inner",
            )

        else:
            features = reduce(
                join, [FIRST_LAST[suite](df) for suite in suites if suite in FIRST_LAST]
            )

    if ("lags" in suites) or ("increase_rate" in suites):
        assert (
            features is not None
        ), f"You should pass one of {list(OPERATIONS.keys())} or {list(FIRST_LAST.keys())} suite."

        features = features.join(
            reduce(
                join,
                [LAG[suite](df, features, options=options) for suite in suites if suite in LAG],
            ),
            on=[df.public_join_key_col, df.public_join_date_col],
            how="inner",
        )

    return features
