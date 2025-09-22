# Databricks notebook source
# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
from datetime import datetime

import mlflow
import numpy as np
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hotel_booking.config import ProjectConfig
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel
from hotel_booking.utils.common import set_mlflow_tracking_uri

# COMMAND ----------
set_mlflow_tracking_uri()
project_config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()
data_loader = DataLoader(spark=spark, config=project_config)
train_query, test_query = data_loader._generate_queries()
train_set = spark.sql(train_query).drop("lead_time", "arrival_month", "repeated", "P_C", "P_not_C", "booking_status")
test_set = spark.sql(test_query).toPandas()

# COMMAND ----------
lead_time_function = f"{project_config.catalog_name}.{project_config.schema_name}.calculate_lead_time"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {lead_time_function}(arrival_date TIMESTAMP, reservation_date TIMESTAMP)
RETURNS BIGINT
LANGUAGE PYTHON AS
$$
return (arrival_date-reservation_date).days
$$""")

arrival_month_function = f"{project_config.catalog_name}.{project_config.schema_name}.get_arrival_month"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {arrival_month_function}(arrival_date TIMESTAMP)
RETURNS BIGINT
LANGUAGE PYTHON AS
$$
return arrival_date.month
$$""")

# COMMAND ----------
feature_table_name = f"{project_config.catalog}.{project_config.schema}.historical_booking_features"
spark.sql(f"""
    CREATE OR REPLACE TABLE {feature_table_name}
    AS SELECT Booking_ID, repeated FROM {project_config.catalog_name}.{project_config.schema_name}.hotel_booking
""")

spark.sql(f"""
    ALTER TABLE {feature_table_name}
    ALTER COLUMN Booking_ID SET NOT NULL
""")

spark.sql(f"""
    ALTER TABLE {feature_table_name}
    ADD CONSTRAINT pk_Booking_ID PRIMARY KEY (Booking_ID)
""")
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
# COMMAND ----------

training_set = fe.create_training_set(
            df=train_set,
            label=project_config.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=feature_table_name,
                    feature_names=["repeated"],
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=lead_time_function,
                    output_name="lead_time",
                    input_bindings={"arrival_date": "arrival_date",
                                    "reservation_date": "date_of_reservation"},
                ),
                FeatureFunction(
                    udf_name=arrival_month_function,
                    output_name="arrival_month",
                    input_bindings={"arrival_date": "arrival_date"},
                ),
            ],
        )

training_df = training_set.load_df().toPandas()
X_train = training_df[project_config.num_features + project_config.cat_features + ["repeated"]]
y_train = training_df[project_config.target]
X_test = test_set[project_config.num_features + project_config.cat_features + ["repeated"]]
y_test = test_set[project_config.target]

# COMMAND ----------
tags = {"branch": "chapter_4", "git_sha": "1234567890abcd"}

model = LightGBMModel(config=project_config)
model.train(
            X_train=X_train,
            y_train=y_train,
        )
mlflow.set_experiment("/Shared/hotel-booking-training-fe")
with mlflow.start_run(run_name=f"lightgbm-training-{datetime.now().strftime('%Y-%m-%d')}",
                description="LightGBM model training",
                tags=tags) as run:
    run_id = run.info.run_id
    mlflow.log_params(model.parameters)
    signature = infer_signature(
        model_input=X_train, model_output=model.pipeline.predict(X_train)
    )
    y_pred = model.pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)

    fe.log_model(
        model=model.pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature)

# COMMAND ----------
model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_model_fe"
registered_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe",
    name=model_name,
    tags=tags,)

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)
# COMMAND ----------
online_store = fe.get_online_store(name="hotel-booking-price-preds")

# Publish the feature table to the online store
fe.publish_table(
    online_store=online_store,
    source_table_name=feature_table_name,
    online_table_name=f"{feature_table_name}_online"
)

# COMMAND ----------
served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=registered_model.version,
    )
]

workspace = WorkspaceClient()
endpoint_name = "hotel-booking-fe"
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
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

serving_endpoint = f"{host}/serving-endpoints/{endpoint_name}/invocations"

payload = {
    "dataframe_split": {
        'columns': [
            'Booking_ID',
            'number_of_adults',
            'number_of_children',
            'number_of_weekend_nights',
            'number_of_week_nights',
            'car_parking_space',
            'special_requests',
            'type_of_meal',
            'room_type',
            'market_segment_type',
            'arrival_month',
            'reservation_date'
        ],
        'data': [['INN36285', 2, 0, 0, 1, 0, 0, 'Not Selected', 'Room_Type 1', 'Online', '2018-08-01 00:00:00', '2018-03-01 00:00:00']]
    }
}

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
)
response.text