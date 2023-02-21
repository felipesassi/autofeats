from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame


@dataclass
class Dataset:
    table: DataFrame
    primary_key_col: str
    table_join_key_col: str
    table_join_date_col: str
    numerical_cols: list
    categorical_cols: list
    public: DataFrame
    public_join_key_col: str
    public_join_date_col: str
    subtract_in_start: int = 0
    subtract_in_end: int = 90
    time_unit: str = "day"

    def select(self):
        self.table = self.table.select(
            self.primary_key_col,
            self.table_join_key_col,
            self.table_join_date_col,
            *self.numerical_cols,
            *self.categorical_cols,
        )

    def __post_init__(self) -> None:
        self.select()

        cnd_1 = self.public[self.public_join_key_col] == self.table[self.table_join_key_col]

        functions = {
            "day": F.date_add,
            "month": F.add_months,
        }

        cnd_2 = self.table[f"{self.table_join_date_col}"] < functions[self.time_unit](
            self.public[self.public_join_date_col], -self.subtract_in_start
        )

        cnd_3 = self.table[f"{self.table_join_date_col}"] > functions[self.time_unit](
            self.public[self.public_join_date_col], -self.subtract_in_end
        )

        cnd = cnd_1 & cnd_2 & cnd_3

        self.or_table = self.table

        self.table = self.public.join(self.table, on=cnd, how="left")
