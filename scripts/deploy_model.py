import mlflow

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.models.pyfunc_model_wrapper import HotelBookingModelWrapper
from hotel_booking.models.serving import serve_model
from hotel_booking.utils.common import create_parser

args = create_parser()

project_config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)


wrapped_model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

wrapped_model_info = mlflow.get_model_version(
    name=f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic",
    version=wrapped_model_version)

catalog = project_config.catalog_name
schema = project_config.schema_name

model_name = f"{catalog}.{schema}.hotel_booking_pyfunc"
wrapper = HotelBookingModelWrapper()

tags = Tags(**{"git_sha": args.git_sha, "branch": args.branch, "run_id": args.run_id})
model_version = wrapper.log_register_model(wrapped_model_info=wrapped_model_info,
                          pyfunc_model_name=model_name,
                          experiment_name="/Shared/hotel-booking-pyfunc",
                          run_id=args.run_id,
                          tags=tags)

serve_model(entity_name=model_name,
            entity_version=model_version,
            tags={"key": "project_name", "value": "hotel_booking"},
            endpoint_name='hotel-booking-pyfunc',
            schema_name=schema,
            catalog_name=catalog,
            table_name_prefix="hotel_booking_monitoring"
            )

