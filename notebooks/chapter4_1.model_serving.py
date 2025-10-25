# Databricks notebook source
import mlflow
import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
from mlflow import MlflowClient
from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

from hotel_booking.config import ProjectConfig

project_config = ProjectConfig.from_yaml("../project_config.yml")

# COMMAND ----------

catalog = project_config.catalog_name
schema = project_config.schema_name

model_name = f"{catalog}.{schema}.hotel_booking_pyfunc"

client = MlflowClient()
model_version = client.get_model_version_by_alias(
    alias="latest-model",
    name=model_name)

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version.version,
    )
]

ai_gateway_cfg = AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True,
            catalog_name=catalog,
            schema_name=schema,
            monitoring_table_suffix="hotel_booking_monitoring",
        )
    )

workspace = WorkspaceClient()
endpoint_name = "hotel-booking-pyfunc"
endpoint_exists = any(
    item.name == endpoint_name for item in workspace.serving_endpoints.list()
)


if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
        ai_gateway=ai_gateway_cfg,
        tags=[EndpointTag.from_dict(
            {"key": "project_name", "value": "hotel_booking"})],
    )
else:
    workspace.serving_endpoints.update_config(
        name=endpoint_name, served_entities=served_entities
    )

# COMMAND ----------
# Call the endpoint
w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

serving_endpoint = f"{host}/serving-endpoints/hotel-booking-pyfunc/invocations"

payload = {
    "dataframe_records": [
        {
            "Booking_ID": "INN1234",
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
        ],
        "data": [["INN1234", 2, 0, 0, 1, 0, 0, 103, "Not Selected", "Room_Type 1", 8, "Online"]],
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
input_example = pd.DataFrame(
    data=payload["dataframe_split"]["data"],
    columns=payload["dataframe_split"]["columns"]
)
model_uri = f"models:/{model_name}@latest-model"
mlflow.models.predict(model_uri, input_example)

# COMMAND ----------
# Convert the input example to serving payload format
serving_payload = convert_input_example_to_serving_input(input_example)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
