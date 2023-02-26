from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame


@dataclass
class Dataset:
    table: DataFrame
    """Dataframe to create features"""
    primary_key_col: str
    """Primary key of the table"""
    table_join_key_col: str
    """Column used to do the join with public"""
    table_join_date_col: str
    """Date column used to do the join with public to create a time window"""
    numerical_cols: list
    """List of numerical columns in the dataframe"""
    categorical_cols: list
    """List of categorical columns in the dataframe"""
    public: DataFrame
    """Public dataframe"""
    public_join_key_col: str
    """Public ID to use in the join"""
    public_join_date_col: str
    """Public date column to use in the join"""
    subtract_in_start: int = 0
    """Days before the reference date"""
    subtract_in_end: int = 90
    """Time window lenght"""
    time_unit: str = "day"
    """Unit used to make time window"""

    def __select(self):
        self.table = self.table.select(
            self.primary_key_col,
            self.table_join_key_col,
            self.table_join_date_col,
            *self.numerical_cols,
            *self.categorical_cols,
        )

    def __post_init__(self) -> None:
        self.__select()

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
