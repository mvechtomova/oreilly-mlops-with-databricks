# Databricks notebook source

import os
from datetime import datetime
from functools import partial

import mlflow
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from hotel_booking.config import ProjectConfig
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel

# COMMAND ----------
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
data_loader = DataLoader(spark=spark, config=project_config)

train_query, valid_query = data_loader.define_split(
    test_months=1,
    train_months=12,
    max_date=datetime.strptime("2018-11-30", "%Y-%m-%d")
)

train_set = spark.sql(train_query).toPandas()
valid_set = spark.sql(valid_query).toPandas()

X_train = train_set[project_config.num_features + project_config.cat_features]
y_train = train_set[project_config.target]

X_valid = valid_set[project_config.num_features + project_config.cat_features]
y_valid = valid_set[project_config.target]

# COMMAND ----------

def train_with_nested_mlflow(config, X_train: pd.DataFrame,
                             X_valid: pd.DataFrame,
                             y_train: pd.DataFrame,
                             y_valid: pd.DataFrame,
                             project_config: ProjectConfig,
                             parent_run_id: str=None):

    n_estimators, max_depth, learning_rate = (
        config["n_estimators"],
        config["max_depth"],
        config["learning_rate"],
    )
    with mlflow.start_run(
        run_name=f"trial_n{n_estimators}_md{max_depth}_lr{learning_rate}", nested=True, parent_run_id=parent_run_id
    ):
        model = LightGBMModel(config=project_config)
        model.train(
            X_train=X_train,
            y_train=y_train,
            parameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
        )
        metrics = model.compute_metrics(X_valid, y_valid)
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        tune.report(metrics)

# COMMAND ----------
mlflow.set_experiment("/Shared/hotel-booking-finetuning")
param_space = {
    "n_estimators": tune.choice([50, 100, 200, 300, 400]),
    "max_depth": tune.choice([3, 5, 10, 15]),
    "learning_rate": tune.choice([0.01, 0.03, 0.05, 0.1, 0.15]),}

run = mlflow.start_run(
    run_name=f"optuna-finetuning-{datetime.now().strftime('%Y-%m-%d')}",
    tags={"git_sha": "1234567890abcd", "branch": "chapter_3"},
    description="LightGBM hyperparameter tuning with Ray & Optuna"
)

trainable = partial(
    train_with_nested_mlflow,
    X_train=X_train,
    X_valid=X_valid,
    y_train=y_train,
    y_valid=y_valid,
    project_config=project_config,
    parent_run_id=run.info.run_id
)

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        search_alg=OptunaSearch(),
        num_samples=50,
        metric="rmse",
        mode="min",
    ),
    param_space=param_space,
)

# COMMAND ----------
# For distributed runs, only on Databricks all-purpose or jobs compute
# import ray
# from ray.util.spark import setup_ray_cluster

# _, remote_conn_str = setup_ray_cluster(num_worker_nodes=2)
# ray.init(remote_conn_str)

results = tuner.fit()
mlflow.end_run()

# COMMAND ----------
best_result=results.get_best_result(metric="rmse", mode="min")
best_result.config

# COMMAND ----------
