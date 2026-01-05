# Databricks notebook source

import time

import pandas as pd
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
# Call endpoint with random records
# First 10 minutes: all data
# Second 10 minutes: append Corporate segment data
# Total duration: 1200s, time per call: 0.3s (0.2 sleep + 0.1 call)
# Total possible calls: 1200 / 0.3 = 4000 (2000 per phase)
duration_per_phase_seconds = 600
sleep_time = 0.2
sample_per_phase = 2000

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

# COMMAND ----------
# Sample data: Phase 1 (all data) + Phase 2 (Corporate segment)
sampled_all = input_data[required_columns].sample(
    n=sample_per_phase, replace=True
)
sampled_corporate = input_data[
    input_data["market_segment_type"] == "Corporate"
][required_columns].sample(n=sample_per_phase, replace=True)

sampled_records = pd.concat([sampled_all, sampled_corporate], ignore_index=True)
logger.info(
    f"Sampled {len(sampled_records)} total records "
    f"({sample_per_phase} all + {sample_per_phase} corporate)"
)

# COMMAND ----------
# Call endpoint with sampled records for 20 minutes
start_time = time.time()
logger.info("Starting endpoint calls for 1200s")

for _, record in sampled_records.iterrows():
    if (time.time() - start_time) >= duration_per_phase_seconds * 2:
        break

    payload = {
        "client_request_id": record["Booking_ID"],
        "dataframe_records": [record.drop("Booking_ID").to_dict()],
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

logger.info("Completed all endpoint calls")

# COMMAND ----------
