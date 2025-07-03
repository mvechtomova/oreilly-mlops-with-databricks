# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.data.data_processor import DataProcessor

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

project_config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = project_config.catalog_name
schema_name = project_config.schema_name

# COMMAND ----------
# Load and process the data
df = pd.read_csv("../data/booking.csv")
data_processor = DataProcessor(df=df, config=project_config, spark=spark)
data_processor.preprocess()
data_processor.generate_synthetic_df(n=1000, max_date=None)
data_processor.save_to_catalog()

# COMMAND ----------
