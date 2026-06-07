# ruff: noqa: F821
import mlflow
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel
from hotel_booking.utils.common import create_parser

args = create_parser()

cfg = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/files/project_config.yml", env=args.env
)

logger.info("Configuration loaded:")
logger.info(yaml.dump(cfg, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Load the dataset
data_loader = DataLoader(spark=spark, config=cfg)
X_train, y_train, X_test, y_test = data_loader.split()

model = LightGBMModel(config=cfg)
tags = Tags(**{"git_sha": args.git_sha, "branch": args.branch, "run_id": args.run_id})
model.train(X_train=X_train, y_train=y_train)
model_info = model.log_model(
    experiment_name="/Shared/hotel-booking-training",
    tags=tags,
    X_test=X_test,
    y_test=y_test,
    train_set_spark=data_loader.train_set_spark,
    train_query=data_loader.train_query,
    test_set_spark=data_loader.test_set_spark,
    test_query=data_loader.test_query,
)

metrics_new = model.metrics

sklearn_model_name = f"{cfg.catalog}.{cfg.schema}.hotel_booking_basic"
eval_data = X_test.copy()
eval_data[cfg.target] = y_test

should_register = True
try:
    mlflow.MlflowClient().get_model_version_by_alias(sklearn_model_name, "latest-model")
    model_exists = True
except mlflow.exceptions.RestException:
    model_exists = False
    logger.info(f"No {sklearn_model_name}@latest-model yet. Registering new model.")

if model_exists:
    result = mlflow.models.evaluate(
        model=f"models:/{sklearn_model_name}@latest-model",
        data=eval_data,
        targets=cfg.target,
        model_type="regressor",
        evaluators=["default"],
    )
    rmse_old = result.metrics["root_mean_squared_error"]
    rmse_new = metrics_new["root_mean_squared_error"]
    should_register = rmse_new < rmse_old
    logger.info(
        f"Current RMSE {rmse_old:.4f}, new RMSE {rmse_new:.4f} -> "
        + ("registering new model" if should_register else "keeping current model")
    )

if should_register:
    model_version = model.register_model(model_name=sklearn_model_name, tags=tags)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
