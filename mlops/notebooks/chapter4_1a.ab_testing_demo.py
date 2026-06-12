# Databricks notebook source
# ruff: noqa

# A/B testing with a pyfunc wrapper. Two model variants are wrapped in one pyfunc
# model that routes each request to A or B based on booking_id. booking_id is NOT
# a feature, so it travels in request params (the only channel that reaches predict).

# COMMAND ----------
from hotel_booking.config import ProjectConfig

cfg = ProjectConfig.from_yaml(
    config_path="../project_config.yml")

model_name_a = f"{cfg.catalog}.{cfg.schema}.hotel_booking_basic"
model_a_uri = f"models:/{model_name_a}@latest-model"

# COMMAND ----------
from pyspark.sql import SparkSession

from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel

spark = SparkSession.builder.getOrCreate()
data_loader = DataLoader(spark=spark, config=cfg)
X_train, y_train, X_test, y_test = data_loader.split()

model_b = LightGBMModel(config=cfg)

model_b.train(
    X_train=X_train,
    y_train=y_train,
    parameters={"learning_rate": 0.01,
                "n_estimators": 1000,
                "max_depth": 6},
)

# COMMAND ----------
from hotel_booking.config import Tags
from hotel_booking.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()
tags = Tags(**{"git_sha": "1234567890abcd", "branch": "main"})

model_b.log_model(
    experiment_name="/Shared/hotel-booking-ab-testing",
    tags=tags,
    X_test=X_test,
    y_test=y_test,
    train_set_spark=data_loader.train_set_spark,
    train_query=data_loader.train_query,
    test_set_spark=data_loader.test_set_spark,
    test_query=data_loader.test_query,
)
model_name_b = f"{cfg.catalog}.{cfg.schema}.hotel_booking_ab_b"

model_b.register_model(model_name=model_name_b, tags=tags)
model_b_uri = f"models:/{model_name_b}@latest-model"

# COMMAND ----------
import hashlib

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext

class HotelBookingABTestWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: PythonModelContext) -> None:
        self.model_a = mlflow.sklearn.load_model(
            context.artifacts["model-a"])
        self.model_b = mlflow.sklearn.load_model(
            context.artifacts["model-b"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame | np.ndarray,
        params: dict | None = None,
    ) -> dict:
        booking_id = str((params or {}).get("booking_id", ""))
        hashed_id = hashlib.md5(booking_id.encode("UTF-8")).hexdigest()
        # Hex digest to int; even/odd gives a stable 50/50 split.
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input)
            variant = "A"
        else:
            predictions = self.model_b.predict(model_input)
            variant = "B"
        prices = [round(float(pred) * 1.05, 2) for pred in predictions]
        return {
            "Total price per night": prices,
            "model": variant,
            "booking_id": booking_id,
        }

# COMMAND ----------
from mlflow.models import infer_signature

# Infer the signature WITH a params block so booking_id is an allowed param with a
# default; without it, schema enforcement drops the param at inference time.
mlflow.set_experiment(experiment_name="/Shared/hotel-booking-ab-testing")
ab_model_name = f"{cfg.catalog}.{cfg.schema}.hotel_booking_ab_test"
wrapper = HotelBookingABTestWrapper()

signature = infer_signature(
    model_input=X_test,
    model_output={
        "Total price per night": [100.00],
        "model": "A",
        "booking_id": "none",
    },
    params={"booking_id": "none"},
)

with mlflow.start_run(run_name="ab-test-wrapper",
                      tags=tags.to_dict()) as run:
    run_id = run.info.run_id
    model_info = mlflow.pyfunc.log_model(
        python_model=wrapper,
        name="pyfunc-ab-test",
        artifacts={"model-a": model_a_uri,
                   "model-b": model_b_uri},
        signature=signature,
        input_example=X_test[0:1],
    )

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=ab_model_name,
    tags=tags.to_dict(),
)

# COMMAND ----------
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=ab_model_name,
    alias="latest-model",
    version=registered_model.version,
)

# COMMAND ----------
# Check routing locally: different booking_id params route to different variants.
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

for booking_id in ["BK-00001", "BK-00002", "BK-00003", "BK-00004"]:
    result = loaded_model.predict(X_test[0:1], params={"booking_id": booking_id})
    print(booking_id, "->", result["model"], result["Total price per night"])

# COMMAND ----------
from mlflow.models import validate_serving_input

# booking_id rides under params, not in dataframe_records. The model parser keeps
# only the records and params keys, so the routing key has to be in params.
serving_payload = {
    "dataframe_records": X_test[0:1].to_dict(orient="records"),
    "params": {"booking_id": "BK-00001"},
}

validate_serving_input(model_info.model_uri, serving_payload)

# COMMAND ----------
from datetime import timedelta

from databricks.sdk import WorkspaceClient

from hotel_booking.models.serving import serve_model

# Deploy both variants behind one endpoint, with an AI Gateway inference table so
# the request payload (params included) is captured. Endpoint creation is async,
# so wait until it is ready before calling it.
endpoint_name = "hotel-booking-ab-test"
serve_model(
    entity_name=ab_model_name,
    entity_version=registered_model.version,
    budget_policy_id=cfg.usage_policy_id,
    endpoint_name=endpoint_name,
    catalog_name=cfg.catalog,
    schema_name=cfg.schema,
    table_name_prefix="hotel_booking_ab_monitoring",
)

w = WorkspaceClient()
w.serving_endpoints.wait_get_serving_endpoint_not_updating(
    name=endpoint_name, timeout=timedelta(minutes=15)
)

# COMMAND ----------
import time

import requests

# Same Booking_ID in two channels: client_request_id for the chapter 6 monitoring
# join (model ignores it), params.booking_id for the A/B routing (reaches predict).
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value
serving_endpoint = f"{host}/serving-endpoints/{endpoint_name}/invocations"

input_data = spark.table(f"{cfg.catalog}.{cfg.schema}.hotel_booking").toPandas()
feature_columns = cfg.num_features + cfg.cat_features
sampled_records = input_data[["Booking_ID", *feature_columns]].sample(n=100, replace=True)

for _, record in sampled_records.iterrows():
    payload = {
        "client_request_id": record["Booking_ID"],
        "dataframe_records": [record.drop("Booking_ID").to_dict()],
        "params": {"booking_id": record["Booking_ID"]},
    }
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    print(response.text)
    time.sleep(0.2)
