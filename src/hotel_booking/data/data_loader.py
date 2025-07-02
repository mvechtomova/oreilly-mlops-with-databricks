from datetime import datetime, timedelta

import pyspark
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig


class DataLoader:
    """A class for loading and splitting the dataset.
    """

    def __init__(self: "DataLoader",
                 config: ProjectConfig,
                 spark: SparkSession) -> None:
        self.config = config
        self.spark = spark

    def define_split(
        self: "DataLoader", test_months: int = 1, train_months: int = 12,
        max_date: datetime = None, version: int = None) -> tuple[str, str]:
        """Split the DataFrame into training and test sets, optionally using a specific Delta table version.

        :param test_months: Number of months for the test set.
        :param train_months: Number of months for the training set.
        :param max_date: The maximum date to consider for splitting.
        :param version: Optional Delta table version to query.
        :return: A tuple containing the training and test SQL queries.
        """
        table_ref = f"{self.config.catalog_name}.{self.config.schema_name}.hotel_booking"

        if max_date is None:
            max_date = self.spark.sql(
                f"SELECT MAX(date_of_reservation) AS max_date FROM {table_ref}"
            ).collect()[0]["max_date"]

        test_set_start = datetime.strftime(max_date.replace(day=1) - relativedelta(months=test_months-1), "%Y-%m-%d")
        test_set_end = datetime.strftime(max_date, "%Y-%m-%d")
        train_set_start = datetime.strftime((max_date.replace(day=1) - relativedelta(months=train_months)).date(), "%Y-%m-%d")
        train_set_end = datetime.strftime((max_date.replace(day=1) - timedelta(days=1)).date(), "%Y-%m-%d")

        version_str = f"VERSION AS OF {version}" if version is not None else ""
        train_query = f"""
            SELECT * FROM {table_ref} {version_str}
            WHERE arrival_date BETWEEN DATE('{train_set_start}') AND DATE('{train_set_end}')
        """
        test_query = f"""
            SELECT * FROM {table_ref} {version_str}
            WHERE arrival_date BETWEEN DATE('{test_set_start}') AND DATE('{test_set_end}')
        """
        return train_query, test_query

    def load_data(self: "DataLoader", train_query: str, test_query: str) -> tuple["pyspark.sql.dataframe.DataFrame", "pyspark.sql.dataframe.DataFrame"]:
        """Load training and testing data from Delta tables as Spark DataFrames."""

        train_set = self.spark.sql(train_query)
        test_set = self.spark.sql(test_query)

        return train_set, test_set
