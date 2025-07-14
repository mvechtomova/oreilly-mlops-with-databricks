# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
from mlflow import MlflowClient

from hotel_booking.config import ProjectConfig

project_config = ProjectConfig.from_yaml("../project_config.yml")

# COMMAND ----------

model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_pyfunc"

client = MlflowClient()
model_version = client.get_model_version_by_alias(alias="latest-model", name=model_name)

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version.version,
    )
]

workspace = WorkspaceClient()
endpoint_name = "hotel-booking-pyfunc"
endpoint_exists = any(item.name ==endpoint_name for item in workspace.serving_endpoints.list())


if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
        tags=[EndpointTag.from_dict({"key": "project_name", "value": "hotel_booking"})]
    )
else:
    workspace.serving_endpoints.update_config(name=endpoint_name, served_entities=served_entities)

# COMMAND ----------
# Call the endpoint
import requests
import pandas as pd

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = context.apiUrl().get()
token = context.apiToken().get()

serving_endpoint = f"{host}/serving-endpoints/hotel-booking-pyfunc/invocations"

payload = {
    "dataframe_records": [{
        'number_of_adults': 2,
        'number_of_children': 0,
        'number_of_weekend_nights': 0,
        'number_of_week_nights': 1,
        'car_parking_space': 0,
        'special_requests': 0,
        'lead_time': 103,
        'type_of_meal': 'Not Selected',
        'room_type': 'Room_Type 1',
        'arrival_month': 8,
        'market_segment_type': 'Online'
    }]
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
        'columns': [
            'number_of_adults',
            'number_of_children',
            'number_of_weekend_nights',
            'number_of_week_nights',
            'car_parking_space',
            'special_requests',
            'lead_time',
            'type_of_meal',
            'room_type',
            'arrival_month',
            'market_segment_type'
        ],
        'data': [[2, 0, 0, 1, 0, 0, 103, 'Not Selected', 'Room_Type 1', 8, 'Online']]
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
import mlflow

input_example = pd.DataFrame(data=payload["dataframe_split"]["data"],
                             columns=payload["dataframe_split"]["columns"])
model_uri = f"models:/{model_name}@latest-model"
mlflow.models.predict(model_uri, input_example)

# COMMAND ----------
from mlflow.models import validate_serving_input
from mlflow.models import convert_input_example_to_serving_input

serving_payload = convert_input_example_to_serving_input(input_example)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
