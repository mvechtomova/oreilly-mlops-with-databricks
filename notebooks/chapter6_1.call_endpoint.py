# Databricks notebook source

import time

import requests
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig

# COMMAND ----------
project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog = project_config.catalog_name
schema = project_config.schema_name

# COMMAND ----------
# Read source table
spark = SparkSession.builder.getOrCreate()
input_data = spark.table(f"{catalog}.{schema}.hotel_booking").toPandas()

logger.info(f"Loaded records from {catalog}.{schema}.hotel_booking")

# COMMAND ----------
# Setup endpoint connection
w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

endpoint_name = "hotel-booking-pyfunc"
serving_endpoint = f"{host}/serving-endpoints/{endpoint_name}/invocations"

# COMMAND ----------
# Call endpoint with random records for 10 minutes
duration_seconds = 600
sleep_time = 0.2
expected_calls = 3000

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
sampled_records = input_data[required_columns].sample(n=expected_calls, replace=True)

logger.info(
    f"Starting endpoint calls for {duration_seconds}s with {sleep_time}s sleep time"
)
logger.info(f"Pre-sampled {expected_calls} records")

start_time = time.time()

for _, record in sampled_records.iterrows():
    if (time.time() - start_time) >= duration_seconds:
        break

    booking_id = record["Booking_ID"]
    features = record.drop("Booking_ID")

    payload = {
        "client_request_id": booking_id,
        "dataframe_records": [features.to_dict()]
    }

    try:
        response = requests.post(
            serving_endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Request failed: {e}")

    time.sleep(sleep_time)

elapsed_time = time.time() - start_time
logger.info("Completed endpoint calling")

# COMMAND ----------
