# Databricks notebook source

import time

import requests
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig

# COMMAND ----------
cfg = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------
# Read source table
spark = SparkSession.builder.getOrCreate()
input_data = spark.table(f"{catalog}.{schema}.hotel_booking").toPandas()

# COMMAND ----------
# Setup endpoint connection
w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

endpoint_name = "hotel-booking-pyfunc"
serving_endpoint = f"{host}/serving-endpoints/{endpoint_name}/invocations"

# COMMAND ----------
# Call endpoint with random records for max 20 minutes
duration_seconds = 1200
sleep_time = 0.2
sample = 6000

# Define required columns for endpoint
required_columns = [
    "Booking_ID",
    "number_of_adults",
    "number_of_children",
    "number_of_weekend_nights",
    "number_of_week_nights",
    "car_parking_space",
    "special_requests",
    "lead_time",
    "type_of_meal",
    "room_type",
    "arrival_month",
    "market_segment_type",
]

# Pre-sample records with only required columns
sampled_records = input_data[required_columns].sample(n=sample, replace=True)
logger.info(f"Pre-sampled {sample} records. Starting calling the endpoint")

# COMMAND ----------
start_time = time.time()
for _, record in sampled_records.iterrows():
    if (time.time() - start_time) >= duration_seconds:
        break

    payload = {
        "client_request_id": record["Booking_ID"],
        "dataframe_records": [record.drop("Booking_ID").to_dict()]
    }

    try:
        response = requests.post(
            serving_endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        print(response.text)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Request failed: {e}")
    time.sleep(sleep_time)
logger.info("Completed endpoint calling")

# COMMAND ----------
