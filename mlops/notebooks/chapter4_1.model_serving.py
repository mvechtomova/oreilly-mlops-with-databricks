# Databricks notebook source

import mlflow

from mlflow import MlflowClient

from hotel_booking.config import ProjectConfig
from hotel_booking.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()

cfg = ProjectConfig.from_yaml("../project_config.yml")

catalog = cfg.catalog
schema = cfg.schema

model_name = "hotel_booking_pyfunc"

client = MlflowClient()
model_version = client.get_model_version_by_alias(
    alias="latest-model",
    name=f"{catalog}.{schema}.{model_name}")

# COMMAND ----------
from databricks.sdk.service.serving import ServedEntityInput

served_entities = [
    ServedEntityInput(
        entity_name=f"{catalog}.{schema}.{model_name}",
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version.version,
    )
]

# COMMAND ----------
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
)
ai_gateway_cfg = AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True,
            catalog_name=catalog,
            schema_name=schema,
            table_name_prefix="hotel_booking_monitoring",
        ),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True)
    )

# COMMAND ----------
from databricks.sdk import WorkspaceClient

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    TrafficConfig,
)
workspace = WorkspaceClient()
endpoint_name = "hotel-booking-pyfunc"
endpoint_exists = any(
    item.name == endpoint_name for item in workspace.serving_endpoints.list()
)

traffic_config=TrafficConfig(
            routes=[
                Route(served_model_name=f"{model_name}-{model_version.version}",
                      traffic_percentage=100)
            ]
        )

if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
            traffic_config=traffic_config),
        ai_gateway=ai_gateway_cfg,
        bugget_policy_id=cfg.usage_policy_id,
    )
else:
    workspace.serving_endpoints.update_config(
        name=endpoint_name, served_entities=served_entities
    )

# COMMAND ----------
# Call the endpoint
import requests

w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

serving_endpoint = f"{host}/serving-endpoints/hotel-booking-pyfunc/invocations"

payload = {
    "dataframe_records": [
        {
            "number_of_adults": 2,
            "number_of_children": 0,
            "number_of_weekend_nights": 0,
            "number_of_week_nights": 1,
            "car_parking_space": 0,
            "special_requests": 0,
            "lead_time": 103,
            "type_of_meal": "Not Selected",
            "room_type": "Room_Type 1",
            "arrival_month": 8,
            "market_segment_type": "Online",
        }
    ]
}

response = requests.post(
    serving_endpoint,
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
)
response.text

# COMMAND ----------
# another way to call the endpoint

payload = {
    "dataframe_split": {
        "columns": [
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
        ],
        "data": [[2, 0, 0, 1, 0, 0, 103, "Not Selected", "Room_Type 1", 8, "Online"]],
    }
}

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
)
response.text

# COMMAND ----------
# For the second payload format (dataframe_split)
import pandas as pd

input_example = pd.DataFrame(
    data=payload["dataframe_split"]["data"],
    columns=payload["dataframe_split"]["columns"]
)
model_uri = f"models:/{model_name}@latest-model"
mlflow.models.predict(model_uri, input_example)

# COMMAND ----------
# Convert the input example to serving payload format
from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

serving_payload = convert_input_example_to_serving_input(input_example)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
