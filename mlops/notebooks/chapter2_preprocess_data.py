# Databricks notebook source
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig

# COMMAND ----------
# Create Spark Session, load configuration
spark = SparkSession.builder.getOrCreate()

cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------
# Create catalog and schema if they do not exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")


# COMMAND ----------
# Load and process the data
df = pd.read_csv("../data/booking.csv")
df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
df["date_of_reservation"] = df["date_of_reservation"].apply(
    lambda x: "3/1/2018" if x == "2018-2-29" else x
)
df["date_of_reservation"] = df["date_of_reservation"].apply(
    lambda x: datetime.strptime(x, "%m/%d/%Y")
)
df["arrival_date"] = df["date_of_reservation"] + pd.to_timedelta(
    df["lead_time"], unit="d"
)
df["arrival_month"] = df["arrival_date"].dt.month

# COMMAND ----------
dst_table = f"{catalog}.{schema}.hotel_booking"

spark.createDataFrame(df).write.mode("append").saveAsTable(f"{dst_table}")

spark.sql(
  f"""ALTER TABLE {dst_table}
   SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"""
)

# COMMAND ----------
