# Databricks notebook source
# Create Spark Session, load configuration
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_booking.config import ProjectConfig

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Load and process the data
df = pd.read_csv("data/booking.csv")
df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
df["date_of_reservation"] = df["date_of_reservation"].apply(lambda x: "3/1/2018" if x == "2018-2-29" else x)
df["date_of_reservation"] = df["date_of_reservation"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
df["arrival_date"] = df["date_of_reservation"] + pd.to_timedelta(df["lead time"], unit="d")
df["booking_status"] = df["booking_status"].replace(["Canceled", "Not_Canceled"], [1, 0])
# only keep market segment "Online"
df = df[df["market_segment_type"] == "Online"]

# COMMAND ----------
# Train and test split, save to UC
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

train_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")

test_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------
spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.train_set " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.test_set " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
