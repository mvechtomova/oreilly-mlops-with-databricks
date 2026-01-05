from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.utils.common import get_delta_table_version


class DataLoader:
    """A class for loading, splitting, and accessing datasets in both Spark and pandas formats."""

    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        """
        :param config: Project configuration containing table and feature info.
        :param spark: Active Spark session.
        """
        self.cfg = config
        self.spark = spark

        self.train_query: str = None
        self.test_query: str  = None
        self.train_set_spark: SparkDataFrame  = None
        self.test_set_spark: SparkDataFrame  = None
        self.train_set: pd.DataFrame  = None
        self.test_set: pd.DataFrame  = None

    def _generate_queries(
        self,
        test_months: int = 1,
        train_months: int = 12,
        max_date: datetime = None,
        version: int = None,
    ) -> tuple[str, str]:
        """
        Generates SQL queries for the training and test splits.

        :param test_months: Number of months for the test set.
        :param train_months: Number of months for the training set.
        :param max_date: The maximum date to consider for splitting. If None, uses the max available date.
        :param version: Optional Delta table version to query.
        :return: A tuple containing the training and test SQL queries.
        """
        table_ref = f"{self.cfg.catalog}.{self.cfg.schema}.hotel_booking"

        if max_date is None:
            max_date = self.spark.sql(
                f"SELECT MAX(date_of_reservation) AS max_date FROM {table_ref}"
            ).collect()[0]["max_date"]

        test_start = (max_date.replace(day=1) - relativedelta(months=test_months - 1)).strftime("%Y-%m-%d")
        test_end = max_date.strftime("%Y-%m-%d")
        train_start = (max_date.replace(day=1) - relativedelta(months=train_months)).strftime("%Y-%m-%d")
        train_end = (max_date.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        if version is None:
            version = get_delta_table_version(self.spark, table_ref)

        train_query = f"""
            SELECT * FROM {table_ref} VERSION AS OF {version}
            WHERE arrival_date BETWEEN DATE('{train_start}') AND DATE('{train_end}')
        """

        test_query = f"""
            SELECT * FROM {table_ref} VERSION AS OF {version}
            WHERE arrival_date BETWEEN DATE('{test_start}') AND DATE('{test_end}')
        """

        return train_query, test_query

    def split(
        self,
        test_months: int = 1,
        train_months: int = 12,
        max_date: datetime = None,
        version: str = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Splits the dataset and loads it into pandas DataFrames for training and testing.

        Also stores Spark and SQL versions of the data internally for inspection or reuse.

        :param test_months: Number of months for the test set.
        :param train_months: Number of months for the training set.
        :param max_date: The maximum date to consider for splitting. If None, the latest date in the table is used.
        :param version: Optional Delta table version to query.
        :return: A tuple (X_train, y_train, X_test, y_test) with pandas DataFrames/Series.
        """
        self.train_query, self.test_query = self._generate_queries(
            test_months, train_months, max_date, version
        )

        self.train_set_spark = self.spark.sql(self.train_query)
        self.test_set_spark = self.spark.sql(self.test_query)

        train_set = self.train_set_spark.toPandas()
        test_set = self.test_set_spark.toPandas()

        X_train = train_set[self.cfg.num_features + self.cfg.cat_features]
        y_train = train_set[self.cfg.target]

        X_test = test_set[self.cfg.num_features + self.cfg.cat_features]
        y_test = test_set[self.cfg.target]

        return X_train, y_train, X_test, y_test
