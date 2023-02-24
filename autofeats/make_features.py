from functools import reduce
from typing import Optional

from pyspark.sql import DataFrame

from autofeats.features.group_by import (
    categorical_statistics,
    correlation_between_features,
    numerical_statistics,
    statistics_of_numerical_data_in_categorical_groups,
)
from autofeats.features.window import (
    first_observation_value,
    lags,
    last_observation_value,
    rate_between_actual_and_past_value,
)
from autofeats.types import Dataset

OPERATIONS = {
    "numerical_statistics": numerical_statistics,
    "numerical_in_categorical_groups": statistics_of_numerical_data_in_categorical_groups,
    "correlation": correlation_between_features,
    "categorical_statistics": categorical_statistics,
}

FIRST_LAST = {
    "first_observation_features": first_observation_value,
    "last_observation_features": last_observation_value,
}

LAG = {
    "lags": lags,
    "increase_rate": rate_between_actual_and_past_value,
}


def run(df: Dataset, suites: list, options: dict) -> Optional[DataFrame]:
    """
    This function will run the feature creation process based on the
    input dataframe, selected features and users'options.

    Each suite will make one type of features:

    - numerical_statistics: numerical statistics (mean, kurtosis, etc) calculated for each numerical column;
    - numerical_in_categorical_groups: numerical statistics (mean, kurtosis, etc) calculated for each numerical column calculated inside each category
    - correlation: correlation between numerical features

    Example::

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

    Args:
        df (Dataset): Dataset with the necessary tables
        suites (list): Suites selected to create features
        options (dict): Options to the suites

    Returns:
        Optional[DataFrame]: Dataframe with features
    """
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
