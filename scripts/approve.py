import argparse

from mlfow import MLflowClient

from hotel_booking.config import ProjectConfig

model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

parser = argparse.ArgumentParser

parser.add_argument("--task_name", action="store", default=None, type=str, required=True)
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default="dev", type=str, required=True)

args = parser.parse_args()

project_config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic"

client = MLflowClient()
tags = client.get_model_version(model_name, model_version).tags

# check if any tag matches the approval tag name
if not any(tag == args.task_name for tag in tags):
  raise Exception("Model version not approved for deployment")
else:
  # if tag is found, check if it is approved
  if tags.get(args.task_name).lower() == "approved":
    print("Model version approved for deployment")
  else:
    raise Exception("Model version not approved for deployment")
