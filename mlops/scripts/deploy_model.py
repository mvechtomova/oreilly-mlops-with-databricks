# ruff: noqa: F821
import mlflow
from loguru import logger

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.models.pyfunc_model_wrapper import HotelBookingModelWrapper
from hotel_booking.models.serving import serve_model
from hotel_booking.utils.common import create_parser

args = create_parser()

model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")
logger.info(f"Retrieved wrapped model version: {model_version}")

cfg = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/files/project_config.yml", env=args.env
)

model_name = f"{cfg.catalog}.{cfg.schema}.hotel_booking_pyfunc"
wrapped_model_info = mlflow.models.get_model_info(
    model_uri=f"models:/{model_name}/{model_version}"
)

wrapper = HotelBookingModelWrapper()

with open(f"{args.root_path}/files/version.txt") as f:
    version = f.read().strip()
whl_name = f"hotel_booking-{version}-py3-none-any.whl"
code_paths = [f"{args.root_path}/artifacts/.internal/{whl_name}"]

tags = Tags(**{"git_sha": args.git_sha, "branch": args.branch, "run_id": args.run_id})

model_version = wrapper.log_register_model(
    wrapped_model_info=wrapped_model_info,
    pyfunc_model_name=model_name,
    experiment_name="/Shared/hotel-booking-pyfunc",
    code_paths=code_paths,
    tags=tags,
)

serve_model(
    entity_name=model_name,
    entity_version=model_version,
    budget_policy_id=cfg.usage_policy_id,
    endpoint_name="hotel-booking-pyfunc",
    schema_name=cfg.schema,
    catalog_name=cfg.catalog,
    table_name_prefix="hotel_booking_monitoring",
)
