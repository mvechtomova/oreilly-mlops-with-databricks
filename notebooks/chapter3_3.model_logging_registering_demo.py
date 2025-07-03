# Databricks notebook source

import json
import os
from datetime import datetime
from importlib.metadata import version

import mlflow
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel

# COMMAND ----------
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
data_loader = DataLoader(spark=spark, config=project_config)

delta_table = DeltaTable.forName(spark, f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking")
latest_version = delta_table.history().select("version").first()[0]
train_query, test_query = data_loader.define_split(version=latest_version)

train_set_spark = spark.sql(train_query)
test_set_spark = spark.sql(test_query)

train_set = train_set_spark.toPandas()
test_set = test_set_spark.toPandas()

X_train = train_set[project_config.num_features + project_config.cat_features]
y_train = train_set[project_config.target]

X_test = test_set[project_config.num_features + project_config.cat_features]
y_test = test_set[project_config.target]


# COMMAND ----------
model = LightGBMModel(config=project_config)
model.train(X_train=X_train,
            y_train=y_train)

mlflow.set_experiment("/Shared/hotel-booking-training")
run = mlflow.start_run(run_name=f"lightgbm-training-{datetime.now().strftime('%Y-%m-%d')}",
                      description="LightGBM model training",
    tags={"git_sha": "1234567890abcd", "branch": "chapter_3"})
run_id = run.info.run_id
metrics = model.compute_metrics(X_test, y_test)

mlflow.log_params(project_config.parameters)
mlflow.log_metrics(metrics)

# COMMAND ----------
signature = infer_signature(
    model_input=X_test, model_output=model.pipeline.predict(X_test)
)
__version__ = version("hotel_booking")
conda_env = _mlflow_conda_env(additional_pip_deps=[f"code/hotel_booking-{__version__}-py3-none-any.whl"])
code_paths = [f"../dist/hotel_booking-{__version__}-py3-none-any.whl"]


# COMMAND ----------
training = mlflow.data.from_spark(
    df=train_set_spark,
    sql=train_query
)
testing = mlflow.data.from_spark(
    df=test_set_spark,
    sql=test_query
)
mlflow.log_input(training, context="training")
mlflow.log_input(testing, context="testing")

# COMMAND ----------
model_info = mlflow.sklearn.log_model(
    sk_model=model.pipeline,
    name="lightgbm-pipeline",
    signature=signature,
    code_paths=code_paths,
    conda_env=conda_env
)
mlflow.end_run()

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
training_input = next((x for x in inputs if x.tags[0].value == 'training'), None)
training_source = mlflow.data.get_source(training_input)
training_source.load()
# COMMAND ----------
testing_input = next((x for x in inputs if x.tags[0].value == 'testing'), None)
testing_source = mlflow.data.get_source(testing_input)
testing_source.load()

# COMMAND ----------
model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic"
registered_model = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=model_name,
            tags={"git_sha": "1234567890abcd", "branch": "chapter_3"},
        )
# COMMAND ----------
client = MlflowClient()

job_id = "1234567890abcdef" # Example job ID; will fail if the job does not exist
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
v = mlflow.search_model_versions(
    filter_string=f"name='{model_name}'")
print(v[0].__dict__)

# COMMAND ----------
# not supported
v = mlflow.search_model_versions(
    filter_string="tags.git_sha='1234567890abcd'")


# COMMAND ----------
# Pyfunc model wrapper
class HotelBookingModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context: PythonModelContext) -> None:
        self.model = mlflow.sklearn.load_model(
            context.artifacts["lightgbm-pipeline"]
        )

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        predictions = self.model.predict(model_input)
        return {"Total price per night": [round(pred*1.05, 2) for pred in predictions]}

wrapped_model = HotelBookingModelWrapper()
sklarn_model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic"
registered_model_name=f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_pyfunc_context"

mlflow.set_experiment(experiment_name="/Shared/hotel-booking-pyfunc")
with mlflow.start_run(run_name=f"wrapper-context-lightgbm-{datetime.now().strftime('%Y-%m-%d')}",
    tags={"branch": "chapter_3",
                            "git_sha": "1234567890abcd"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train,
                                model_output={'Total price per night': [100.00]})
    pyfunc_model = mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        name="pyfunc-wrapper",
        artifacts={
            "lightgbm-pipeline": f"models:/{sklarn_model_name}@latest-model"},
        registered_model_name=registered_model_name,
        code_paths=code_paths,
        conda_env=conda_env,
        signature=signature
    )
# COMMAND ----------
loaded_pufunc_model = mlflow.pyfunc.load_model(pyfunc_model.model_uri)
# COMMAND ----------
client.set_registered_model_alias(
    name=registered_model_name,
    alias="latest-model",
    version=pyfunc_model.registered_model_version,
)
# COMMAND ----------
unwraped_model = loaded_pufunc_model.unwrap_python_model()
unwraped_model.predict(context=None, model_input=X_test[0:1])
