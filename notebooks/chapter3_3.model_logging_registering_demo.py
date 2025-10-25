# Databricks notebook source

import json
from datetime import datetime

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel
from hotel_booking.utils.common import set_mlflow_tracking_uri

# COMMAND ----------
set_mlflow_tracking_uri()
project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")
tags = Tags(**{"git_sha": "1234567890abcd", "branch": "main"})

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
data_loader = DataLoader(spark=spark, config=project_config)
X_train, y_train, X_test, y_test = data_loader.split()

# COMMAND ----------
model = LightGBMModel(config=project_config)
model.train(X_train=X_train, y_train=y_train)

mlflow.set_experiment("/Shared/hotel-booking-training")
run = mlflow.start_run(
    run_name=f"lightgbm-training-{datetime.now().strftime('%Y-%m-%d')}",
    description="LightGBM model training",
    tags=tags.to_dict(),
)
run_id = run.info.run_id

mlflow.log_params(project_config.parameters)

# COMMAND ----------
signature = infer_signature(
    model_input=X_test, model_output=model.pipeline.predict(X_test)
)

# COMMAND ----------
training = mlflow.data.from_spark(
    df=data_loader.train_set_spark, sql=data_loader.train_query
)
testing = mlflow.data.from_spark(
    df=data_loader.test_set_spark, sql=data_loader.test_query
)
mlflow.log_input(training, context="training")
mlflow.log_input(testing, context="testing")

# COMMAND ----------
model_info = mlflow.sklearn.log_model(
    sk_model=model.pipeline,
    name="lightgbm-pipeline",
    signature=signature,
    input_example=X_test[0:1],
)
eval_data = X_test.copy()
eval_data[project_config.target] = y_test

# This will log the evaluation metrics
result = mlflow.models.evaluate(
    model_info.model_uri,
    eval_data,
    targets=project_config.target,
    model_type="regressor",
    evaluators=["default"],
)
mlflow.end_run()

# COMMAND ----------
result.metrics

# COMMAND ----------
logged_model = mlflow.get_logged_model(model_info.model_id)
model = mlflow.sklearn.load_model(f"models:/{model_info.model_id}")
# COMMAND ----------
logged_model_dict = logged_model.to_dictionary()
logged_model_dict["metrics"] = [x.__dict__ for x in logged_model_dict["metrics"]]
with open("../demo_artifacts/logged_model.json", "w") as json_file:
    json.dump(logged_model_dict, json_file, indent=4)
# COMMAND ----------
logged_model.params
# COMMAND ----------
logged_model.metrics
# COMMAND ----------
run = mlflow.get_run(run_id)

# COMMAND ----------
inputs = run.inputs.dataset_inputs
training_input = next(
    (x for x in inputs if len(x.tags) > 0 and x.tags[0].value == "training"), None
)
training_source = mlflow.data.get_source(training_input)
training_source.load()
# COMMAND ----------
testing_input = next(
    (x for x in inputs if len(x.tags) > 0 and x.tags[0].value == "testing"), None
)
testing_source = mlflow.data.get_source(testing_input)
testing_source.load()

# COMMAND ----------
model_name = (
    f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic"
)
registered_model = mlflow.register_model(
    model_uri=logged_model.model_uri,
    name=model_name,
    tags=tags.to_dict(),
)
# COMMAND ----------
client = MlflowClient()

job_id = "1234567890abcdef"  # Example job ID; will fail if the job does not exist
client.create_registered_model(model_name, deployment_job_id=job_id)

# COMMAND ----------
# latest alias is reserved, so we cannot use it
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)
model_version = client.get_model_version_by_alias(alias="latest-model", name=model_name)
model = mlflow.sklearn.load_model(f"models:/{model_name}@latest-model")

# COMMAND ----------
# only searching by name is supported
v = mlflow.search_model_versions(filter_string=f"name='{model_name}'")
print(v[0].__dict__)

# COMMAND ----------
# not supported
v = mlflow.search_model_versions(filter_string="tags.git_sha='1234567890abcd'")


# COMMAND ----------
# Pyfunc model wrapper
from importlib.metadata import version
from hotel_booking.models.pyfunc_model_wrapper import HotelBookingModelWrapper

__version__ = version("hotel_booking")
code_paths = [f"../dist/hotel_booking-{__version__}-py3-none-any.whl"]

wrapper = HotelBookingModelWrapper()
pyfunc_model_name = (
    f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_pyfunc"
)

wrapper.log_register_model(
    wrapped_model_info=model_info,
    pyfunc_model_name=pyfunc_model_name,
    experiment_name="/Shared/hotel-booking-wrapper",
    tags=tags,
    code_paths=code_paths,
)


# COMMAND ----------
loaded_pufunc_model = mlflow.pyfunc.load_model(wrapper.model_info.model_uri)
# COMMAND ----------
unwraped_model = loaded_pufunc_model.unwrap_python_model()
unwraped_model.predict(context=None, model_input=X_test[0:1])
