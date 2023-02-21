import pandas as pd
import pandas.testing as pd_test
import pyspark.sql.functions as F

from autofeats.features import group_by
from tests.utils import features, make_dataset, make_dataset_to_correlation, spark


def test_correlation_between_features_expression(
    make_dataset_to_correlation,
):
    features = group_by.correlation_between_features(make_dataset_to_correlation)

    assert str(features) == str(
        [F.corr("paid_value", "discount").alias("corr_between___paid_value_discount")]
    )


def test_correlation_between_features_values(make_dataset_to_correlation):
    features = group_by.correlation_between_features(make_dataset_to_correlation)

    results = (
        make_dataset_to_correlation.table.groupby(
            make_dataset_to_correlation.public_join_key_col,
            make_dataset_to_correlation.public_join_date_col,
        )
        .agg(*features)
        .toPandas()
    )

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 2, 3],
            "date_ref": ["2022-01-04", "2022-01-04", "2022-01-04"],
            "corr_between___paid_value_discount": [1.0, 1.0, 1.0],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None


def test_get_categories_from_categorical_data(make_dataset):
    categories = group_by.get_categories_from_categorical_data(make_dataset)

    assert categories == [{"values": ["C", "B", "A"], "col_name": "product_type"}]


def test_numerical_statistics_expression(make_dataset):
    features = group_by.numerical_statistics(make_dataset)

    assert (
        str(features)
        == "[Column<'sum(paid_value) AS sum___paid_value'>, Column<'avg(paid_value) AS mean___paid_value'>, Column<'stddev_samp(paid_value) AS stddev___paid_value'>, Column<'min(paid_value) AS min___paid_value'>, Column<'max(paid_value) AS max___paid_value'>, Column<'kurtosis(paid_value) AS kurtosis___paid_value'>, Column<'skewness(paid_value) AS skewness___paid_value'>]"
    )


def test_numerical_statistics_values(make_dataset):
    features = group_by.numerical_statistics(make_dataset)

    results = (
        make_dataset.table.groupby(
            make_dataset.public_join_key_col, make_dataset.public_join_date_col
        )
        .agg(*features)
        .toPandas()
    )

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 2, 3],
            "date_ref": ["2022-01-04", "2022-01-04", "2022-01-04"],
            "sum___paid_value": [28, 36, 45],
            "mean___paid_value": [9.333333, 12.0, 15.0],
            "stddev___paid_value": [8.504901, 9.0, 9.0],
            "min___paid_value": [1, 3, 6],
            "max___paid_value": [18, 21, 24],
            "kurtosis___paid_value": [-1.5, -1.5, -1.5],
            "skewness___paid_value": [0.071892, 0.0, 0.0],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None


def test_categorical_statistics_expression(make_dataset):
    features = group_by.categorical_statistics(make_dataset)

    assert (
        str(features)
        == "[Column<'count(CASE WHEN (product_type = C) THEN product_type END) AS `product_type=C__count___product_type`'>, Column<'count(CASE WHEN (product_type = B) THEN product_type END) AS `product_type=B__count___product_type`'>, Column<'count(CASE WHEN (product_type = A) THEN product_type END) AS `product_type=A__count___product_type`'>, Column<'count(product_type) AS count___product_type'>, Column<'count(product_type) AS countDistinct___product_type'>]"
    )


def test_categorical_statistics_values(make_dataset):
    features = group_by.categorical_statistics(make_dataset)

    results = make_dataset.table.groupby("consumer_id_ref", "date_ref").agg(*features).toPandas()

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 2, 3],
            "date_ref": ["2022-01-04", "2022-01-04", "2022-01-04"],
            "product_type=C__count___product_type": [1, 1, 1],
            "product_type=B__count___product_type": [1, 1, 1],
            "product_type=A__count___product_type": [1, 1, 1],
            "count___product_type": [3, 3, 3],
            "countDistinct___product_type": [3, 3, 3],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None
