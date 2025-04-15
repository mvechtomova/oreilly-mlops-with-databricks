# Databricks notebook source
from datetime import datetime, timezone

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

# COMMAND ----------
# Create Spark session, load configuration
spark = SparkSession.builder.getOrCreate()

with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

# COMMAND ----------
# Load and process the data
df = pd.read_csv("data/booking.csv")
df["date of reservation"] = df["date of reservation"].apply(lambda x: "3/1/2018" if x == "2018-2-29" else x)
df["reservation_date"] = df["date of reservation"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
df["arrival_date"] = df["reservation_date"] + pd.to_timedelta(df["lead time"], unit="d")
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("-", "_")

df["booking_status"] = df["booking_status"].replace(["Canceled", "Not_Canceled"], [1, 0])
df = df.drop(columns=["date_of_reservation"])
extra_set = df[df["market_segment_type"] != "Online"]
df = df[df["market_segment_type"] == "Online"]

# COMMAND ----------
# Train and test split, save to UC
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

spark.createDataFrame(extra_set).write.saveAsTable(f"{catalog_name}.{schema_name}.extra_set")

spark.createDataFrame(test_set).write.saveAsTable(f"{catalog_name}.{schema_name}.test_set")

spark.createDataFrame(test_set).write.saveAsTable(f"{catalog_name}.{schema_name}.train_set")

# COMMAND ----------
for table_name in ["train_set", "test_set"]:
    spark.sql(
        f"ALTER TABLE {catalog_name}.{schema_name}.{table_name} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    )

    spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.{table_name} " "ADD COLUMN update_timestamp_utc TIMESTAMP;")

    time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    spark.sql(f"UPDATE {catalog_name}.{schema_name}.{table_name} " f"SET update_timestamp_utc = '{time_now}';")
