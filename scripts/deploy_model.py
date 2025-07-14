from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
import argparse
import mlflow

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.models.pyfunc_model_wrapper import HotelBookingModelWrapper

parser = argparse.ArgumentParser

parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default="dev", type=str, required=True)
parser.add_argument("--branch", action="store", default="dev", type=str, required=True)
parser.add_argument("--git_sha", action="store", default="dev", type=str, required=True)
parser.add_argument("--run_id", action="store", default="dev", type=str, required=True)

args = parser.parse_args()

project_config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
tags = Tags(**{"git_sha": args.git_sha, "branch": args.branch, "run_id": args.run_id})

wrapped_model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

wrapped_model_info = mlflow.get_model_version(
    name=f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic",
    version=wrapped_model_version)

model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_pyfunc"

wrapper = HotelBookingModelWrapper()

model_version = wrapper.log_register_model(wrapped_model_uri=wrapped_model_info.model_uri,
                          pyfunc_model_name=model_name,
                          experiment_name="/Shared/hotel-booking-pyfunc",
                          tags=tags)

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version,
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
