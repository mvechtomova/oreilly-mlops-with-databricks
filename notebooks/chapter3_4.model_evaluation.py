# Databricks notebook source
# MAGIC %pip install -e ..
# COMMAND ----------
# MAGIC %restart_python
# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))
# COMMAND ----------
import mlflow
import pandas as pd
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.data import DataLoader, DataProcessor
from hotel_booking.models.lightgbm_model import LightGBMModel
from hotel_booking.utils.common import set_mlflow_tracking_uri

# COMMAND ----------
set_mlflow_tracking_uri()
spark = SparkSession.builder.getOrCreate()

cfg = ProjectConfig.from_yaml("../project_config.yml")
# COMMAND ----------
# Load and process the data
df = pd.read_csv("../data/booking.csv")
data_processor = DataProcessor(df=df, config=cfg, spark=spark)
data_processor.preprocess()
data_processor.generate_synthetic_df(n=1000, max_date=None)
data_processor.save_to_catalog()

# COMMAND ----------
data_loader = DataLoader(spark=spark, config=cfg)
X_train, y_train, X_test, y_test = data_loader.split()

# COMMAND ----------
# retrain model
model = LightGBMModel(config=cfg)
tags=Tags(**{"git_sha": "1234567890abcd", "branch": "chapter_3"})
model.train(X_train=X_train,
            y_train=y_train)
model_info = model.log_model(
    experiment_name="/Shared/hotel-booking-training",
    tags=tags,
    X_test=X_test,
    y_test=y_test,
    train_set_spark=data_loader.train_set_spark,
    train_query=data_loader.train_query,
    test_set_spark=data_loader.test_set_spark,
    test_query=data_loader.test_query
    )

metrics_new = model.metrics
# COMMAND ----------
# eval model

sklearn_model_name = f"{cfg.catalog}.{cfg.schema}.hotel_booking_basic"
model_uri = f"models:/{sklearn_model_name}@latest-model"
eval_data = X_test.copy()
eval_data[cfg.target] = y_test

result = mlflow.models.evaluate(
        model_uri,
        eval_data,
        targets=cfg.target,
        model_type="regressor",
        evaluators=["default"],
    )
metrics_old = result.metrics

# COMMAND ----------
if metrics_new['root_mean_squared_error'] < metrics_old['root_mean_squared_error']:
    model.register_model(model_name=sklearn_model_name, job_id=None, tags=tags)
