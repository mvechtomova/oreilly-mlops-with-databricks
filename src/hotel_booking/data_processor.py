"""Data preprocessing module."""

import datetime

import pandas as pd
from hotel_booking.config import ProjectConfig
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to UC tables.
    """

    def __init__(self: "DataProcessor", pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self: "DataProcessor") -> None:
        """Preprocess the DataFrame stored in self.df."""
        self.df.columns = self.df.columns.str.replace(r"[ -]", "_", regex=True)
        self.df["date_of_reservation"] = self.df["date_of_reservation"].apply(
            lambda x: "3/1/2018" if x == "2018-2-29" else x
        )
        self.df["date_of_reservation"] = self.df["date_of_reservation"].apply(
            lambda x: datetime.strptime(x, "%m/%d/%Y")
        )
        self.df["arrival_date"] = self.df["date_of_reservation"] + pd.to_timedelta(self.df["lead_time"], unit="d")
        self.df["booking_status"] = self.df["booking_status"].replace(["Canceled", "Not_Canceled"], [1, 0])
        # only keep market segment "Online"
        self.df = self.df[self.df["market_segment_type"] == "Online"]

    def split_data(
        self: "DataProcessor", test_size: float = 0.2, random_state: int = 28
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self: "DataProcessor", train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self: "DataProcessor") -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
