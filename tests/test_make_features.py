import pytest

from autofeat import make_features
from tests.utils import features, make_dataset, make_dataset_to_correlation, spark


@pytest.mark.parametrize("suites", [(["lags"]), (["increase_rate"])])
def test_make_features_assertion(make_dataset, suites):
    with pytest.raises(AssertionError):
        make_features.run(
            df=make_dataset,
            suites=suites,
            options={"n_lags": [1, 3, 4, 5]},
        )


@pytest.mark.parametrize(
    "suites",
    [
        (["numerical_statistics"]),
        (["numerical_statistics", "lags"]),
        (["numerical_statistics", "increase_rate"]),
        (["numerical_in_categorical_groups"]),
        (["categorical_statistics"]),
        (["first_observation_features"]),
        (["last_observation_features"]),
    ],
)
def test_make_features(make_dataset, suites):
    features = make_features.run(
        df=make_dataset,
        suites=suites,
        options={"n_lags": [1, 3, 4, 5]},
    )

    assert features is not None
