# Databricks notebook source

# % pip install -e ..
# COMMAND ----------
# %restart_python
# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig

# COMMAND ----------
# Create Spark Session, load configuration
spark = SparkSession.builder.getOrCreate()

project_config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = project_config.catalog_name
schema_name = project_config.schema_name

# COMMAND ----------
# Load and process the data
df = pd.read_csv("../data/booking.csv")
df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
df["date_of_reservation"] = df["date_of_reservation"].apply(lambda x: "3/1/2018" if x == "2018-2-29" else x)
df["date_of_reservation"] = df["date_of_reservation"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
df["arrival_date"] = df["date_of_reservation"] + pd.to_timedelta(df["lead_time"], unit="d")
df["arrival_month"] = df["arrival_date"].dt.month

# COMMAND ----------
spark.createDataFrame(df).write.mode("append").saveAsTable(
   f"{catalog_name}.{schema_name}.hotel_booking")
spark.sql(
   f"ALTER TABLE {catalog_name}.{schema_name}.hotel_booking "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
# COMMAND ----------
