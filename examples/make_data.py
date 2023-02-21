import numpy as np
import pandas as pd


def make_transactions(n=10000):
    return pd.DataFrame(
        data={
            "transaction_id": range(n),
            "consumer_id": [np.random.choice(list(range(100))) for _ in range(n)],
            "paid_value": np.random.normal(1000, 100, n),
            "discount": [np.random.choice([10, 0, 25]) for _ in range(n)],
            "product_type": [
                str(np.random.choice(["a", "b", "c", "d", "e", "f", "g", "h"])) for _ in range(n)
            ],
            "buy_type": [
                str(
                    np.random.choice(
                        ["f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9"]
                    )
                )
                for _ in range(n)
            ],
            "paymnent_date": pd.date_range("2022-01-01", "2023-01-01", periods=n),
        }
    )
