import datetime

import pandas as pd
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing and saving to UC tables.
    """

    def __init__(self: "DataProcessor",
                 config: ProjectConfig,
                 spark: SparkSession) -> None:
        self.config = config
        self.spark = spark

    def preprocess_and_save_to_catalog(df: pd.DataFrame, self: "DataProcessor") -> None:
        """Preprocess the DataFrame stored in self.df."""
        df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
        # remove all self.df and replace with df
        df["date_of_reservation"] = df["date_of_reservation"].apply(
            lambda x: "3/1/2018" if x == "2018-2-29" else x
        )
        df["date_of_reservation"] = df["date_of_reservation"].apply(
            lambda x: datetime.strptime(x, "%m/%d/%Y")
        )
        df["arrival_date"] = df["date_of_reservation"] + pd.to_timedelta(df["lead_time"], unit="d")
        df["arrival_month"] = df["arrival_date"].dt.month

        self.spark.createDataFrame(df).write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.hotel_booking")
        self.spark.sql(
        f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.hotel_booking "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
