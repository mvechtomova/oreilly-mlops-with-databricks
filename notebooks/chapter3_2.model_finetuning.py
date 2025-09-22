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
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel
from hotel_booking.utils.common import set_mlflow_tracking_uri

# COMMAND ----------
set_mlflow_tracking_uri()
project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
data_loader = DataLoader(spark=spark, config=project_config)

X_train, y_train, X_valid, y_valid = data_loader.split(
    test_months=1,
    train_months=12,
    max_date=datetime.strptime("2018-11-30", "%Y-%m-%d")
)

# COMMAND ----------

param_space = {
    "n_estimators": tune.qrandint(50, 700, q=50),
    "learning_rate": tune.loguniform(0.01, 0.2),
    "max_depth": tune.choice([3, 5, 10, 15])}


# COMMAND ----------
def train_with_nested_mlflow(config, X_train: pd.DataFrame,
                             X_valid: pd.DataFrame,
                             y_train: pd.DataFrame,
                             y_valid: pd.DataFrame,
                             project_config: ProjectConfig,
                             experiment_id: str,
                             parent_run_id: str):

    n_estimators, max_depth, learning_rate = (
        config["n_estimators"],
        config["max_depth"],
        config["learning_rate"],
    )
    with mlflow.start_run(experiment_id=experiment_id,
        run_name=f"trial_n{n_estimators}_md{max_depth}_lr{learning_rate}",
        nested=True, parent_run_id=parent_run_id
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
        y_pred = model.pipeline.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = np.sqrt(mse)
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mean_absolute_error(y_valid, y_pred),
            "r2_score": r2_score(y_valid, y_pred),
        }
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        tune.report(metrics)

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from ray.util.spark import setup_ray_cluster

w = WorkspaceClient()
os.environ["DATABRICKS_HOST"] = w.config.host
os.environ["DATABRICKS_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# for distributed, use this:
ray_conf = setup_ray_cluster(
  min_worker_nodes=2,
  max_worker_nodes=8,
)
os.environ['RAY_ADDRESS'] = ray_conf[0]
# COMMAND ----------

mlflow.set_experiment("/Shared/hotel-booking-finetuning")
experiment_id = mlflow.get_experiment_by_name("/Shared/hotel-booking-finetuning").experiment_id

n_trials = 50

with mlflow.start_run(
    run_name=f"optuna-finetuning-{datetime.now().strftime('%Y-%m-%d')}",
    tags={"git_sha": "1234567890abcd", "branch": "main"},
    description="LightGBM hyperparameter tuning with Ray & Optuna"
) as parent_run:

    tuner = tune.Tuner(
        tune.with_parameters(
                train_with_nested_mlflow,
                X_train_in = X_train,
                y_train_in = y_train,
                X_valid_in = X_valid,
                y_valid_in = y_valid,
                project_config=project_config,
                parent_run_id=parent_run.info.run_id,
                experiment_id=experiment_id,
            ),
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(metric="rmse", mode="min"),
            num_samples=n_trials,
        ),
        param_space=param_space,
    )
    results = tuner.fit()

# COMMAND ----------
best_result=results.get_best_result(metric="rmse", mode="min")
best_result.config

# COMMAND ----------
