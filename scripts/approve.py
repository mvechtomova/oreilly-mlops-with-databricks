from loguru import logger
from mlflow import MLflowClient

from hotel_booking.config import ProjectConfig
from hotel_booking.utils.common import create_parser
from hotel_booking.utils.exceptions import DeploymentNotApprovedError, MissingDeploymentTagError

model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

args = create_parser()

project_config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic"

client = MLflowClient()
tags = client.get_model_version(model_name, model_version).tags

# check if any tag matches the approval tag name
if not any(tag == args.task_name for tag in tags):
  raise MissingDeploymentTagError("Model version not approved for deployment")
else:
  # if tag is found, check if it is approved
  if tags.get(args.task_name).lower() == "approved":
    logger.info("Model version approved for deployment")
  else:
    raise DeploymentNotApprovedError("Model version not approved for deployment")
