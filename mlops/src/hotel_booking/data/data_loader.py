from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.utils.common import get_delta_table_version


class DataLoader:
    """Loads, splits, and accesses datasets in both Spark and pandas formats."""

    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        """
        :param config: Project configuration containing table and feature info.
        :param spark: Active Spark session.
        """
        self.cfg = config
        self.spark = spark

    def split(
        self,
        test_months: int = 1,
        train_months: int = 12,
        max_date: datetime = None,
        version: str = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Splits the dataset and loads it into pandas DataFrames for training and testing.

        :param test_months: Number of months for the test set.
        :param train_months: Number of months for the training set.
        :param max_date: The maximum date for splitting. If None, uses the latest date.
        :param version: Optional Delta table version to query.
        :return: A tuple (X_train, y_train, X_test, y_test) with pandas DataFramess.
        """
        self.train_query, self.test_query = self._generate_queries(
            test_months, train_months, max_date, version)

        self.train_set_spark = self.spark.sql(self.train_query)
        self.test_set_spark = self.spark.sql(self.test_query)

        train_set = self.train_set_spark.toPandas()
        test_set = self.test_set_spark.toPandas()

        X_train = train_set[self.cfg.num_features + self.cfg.cat_features]
        y_train = train_set[self.cfg.target]

        X_test = test_set[self.cfg.num_features + self.cfg.cat_features]
        y_test = test_set[self.cfg.target]

        return X_train, y_train, X_test, y_test

    def _get_start_end_dates(
        self,
        max_date: datetime,
        test_months: int,
        train_months: int,
    ) -> tuple[str, str, str, str]:
        """
        Computes the start and end dates for the train and test splits.

        :param max_date: The maximum date in the dataset.
        :param test_months: Number of months for the test set.
        :param train_months: Number of months for the training set.
        :return: A tuple (train_start, train_end, test_start, test_end).
        """
        fmt = "%Y-%m-%d"
        month_start = max_date.replace(day=1)
        train_start = (month_start - relativedelta(months=train_months)).strftime(fmt)
        train_end = (month_start - timedelta(days=1)).strftime(fmt)
        test_start = (month_start - relativedelta(months=test_months - 1)).strftime(fmt)
        test_end = max_date.strftime(fmt)
        return train_start, train_end, test_start, test_end

    def _generate_queries(
        self, test_months: int = 1, train_months: int = 12,
        max_date: datetime = None, version: int = None,
    ) -> tuple[str, str]:
        """
        Generates SQL queries for the training and test splits.

        :param test_months: Number of months for the test set.
        :param train_months: Number of months for the training set.
        :param max_date: The maximum date for splitting. If None, uses the max date.
        :param version: Optional Delta table version to query.
        :return: A tuple containing the training and test SQL queries.
        """
        table_ref = f"{self.cfg.catalog}.{self.cfg.schema}.hotel_booking"

        if max_date is None:
            max_date = self.spark.sql(
                f"SELECT MAX(date_of_reservation) AS max_date FROM {table_ref}"
            ).collect()[0]["max_date"]

        if version is None:
            version = get_delta_table_version(self.spark, table_ref)

        train_start, train_end, test_start, test_end = self._get_start_end_dates(
            max_date, test_months, train_months)

        train_query = f"""
            SELECT * FROM {table_ref} VERSION AS OF {version}
            WHERE arrival_date BETWEEN DATE('{train_start}') AND DATE('{train_end}')
        """

        test_query = f"""
            SELECT * FROM {table_ref} VERSION AS OF {version}
            WHERE arrival_date BETWEEN DATE('{test_start}') AND DATE('{test_end}')
        """

        return train_query, test_query
