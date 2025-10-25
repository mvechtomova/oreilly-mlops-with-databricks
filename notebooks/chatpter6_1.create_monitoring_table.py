# Databricks notebook source

import os
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hotel_booking.config import ProjectConfig


# COMMAND ----------
project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog = project_config.catalog_name
schema = project_config.schema_name

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
input_data = spark.table(f"{catalog}.{schema}.hotel_booking").toPandas()

