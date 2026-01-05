import calendar
import datetime
import random
from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from hotel_booking.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing and saving to UC tables.
    """

    def __init__(
        self: "DataProcessor",
        df: pd.DataFrame,
        config: ProjectConfig,
        spark: SparkSession,
    ) -> None:
        self.df = df
        self.cfg = config
        self.spark = spark
        self.target_table = (
            f"{self.cfg.catalog}.{self.cfg.schema}.hotel_booking"
        )

    def preprocess(self: "DataProcessor") -> None:
        """Preprocess the DataFrame."""
        self.df.columns = self.df.columns.str.replace(r"[ -]", "_", regex=True)
        self.df["date_of_reservation"] = self.df["date_of_reservation"].apply(
            lambda x: "3/1/2018" if x == "2018-2-29" else x
        )
        self.df["date_of_reservation"] = self.df["date_of_reservation"].apply(
            lambda x: datetime.strptime(x, "%m/%d/%Y")
        )
        self.df["arrival_date"] = self.df["date_of_reservation"] + pd.to_timedelta(
            self.df["lead_time"], unit="d"
        )
        self.df["arrival_month"] = self.df["arrival_date"].dt.month

    def generate_synthetic_df(
        self: "DataProcessor", n: int = 1000, max_date: datetime = None
    ) -> None:
        if max_date is None:
            max_date = self.spark.sql(
                f"SELECT MAX(date_of_reservation) AS max_date FROM {self.target_table}"
            ).collect()[0]["max_date"]
        start_date_to_polulate = max_date.replace(day=1) + relativedelta(months=1)
        year = start_date_to_polulate.year
        month = start_date_to_polulate.month

        metadata = Metadata.detect_from_dataframe(self.df)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data=self.df)
        days_in_month = calendar.monthrange(year, month)[1]
        start_date = datetime(year, month, 1)
        all_dates = [
            (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days_in_month)
        ]
        self.df = synthesizer.sample(num_rows=n)
        self.df["date_of_reservation"] = [random.choice(all_dates) for _ in range(n)]
        self.df["date_of_reservation"] = pd.to_datetime(self.df["date_of_reservation"])
        self.df["arrival_date"] = self.df["date_of_reservation"] + pd.to_timedelta(
            self.df["lead_time"], unit="d"
        )
        self.df["arrival_month"] = pd.to_datetime(self.df["arrival_date"]).dt.month

    def save_to_catalog(self: "DataProcessor") -> None:
        """Preprocess the DataFrame stored in self.df."""
        self.spark.createDataFrame(self.df).write.mode("append").saveAsTable(
            f"{self.target_table}"
        )
        self.spark.sql(
            f"ALTER TABLE {self.target_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
