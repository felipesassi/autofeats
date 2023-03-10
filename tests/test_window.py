import numpy as np
import pandas as pd
import pandas.testing as pd_test

from autofeats.features import window
from tests.utils import features, make_dataset, spark


def test_last_observation_value(make_dataset):
    results = window.last_observation_value(make_dataset).toPandas()

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 2, 3],
            "date_ref": ["2022-01-04", "2022-01-04", "2022-01-04"],
            "last___paid_value": [18, 21, 24],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None


def test_first_observation_value(make_dataset):
    results = window.first_observation_value(make_dataset).toPandas()

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 2, 3],
            "date_ref": ["2022-01-04", "2022-01-04", "2022-01-04"],
            "first___paid_value": [1, 3, 6],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None


def test_rate_between_actual_and_past_value(make_dataset, features):
    results = window.rate_between_actual_and_past_value(make_dataset, features).toPandas()

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 1, 1],
            "date_ref": ["2022-01-04", "2022-01-05", "2022-01-06"],
            "increase_rate_first___paid_value": [np.nan, 2.0, 1.0],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None


def test_lags(make_dataset, features):
    results = window.lags(make_dataset, features).toPandas()

    baseline = pd.DataFrame(
        data={
            "consumer_id_ref": [1, 1, 1],
            "date_ref": ["2022-01-04", "2022-01-05", "2022-01-06"],
            "lag=1_first___paid_value": [np.nan, 1.0, 3.0],
        }
    )

    assert pd_test.assert_frame_equal(results, baseline) is None
