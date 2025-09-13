# Databricks notebook source
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from hotel_booking.config import ProjectConfig

# COMMAND ----------
project_config = ProjectConfig.from_yaml("../project_config.yml")
spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()

# COMMAND ----------
model_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_basic"
model_uri = f"models:/{model_name}@latest-model"
columns = project_config.cat_features + project_config.num_features

input_df = spark.table(f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking")
model_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

preds_df = input_df.select(
    col("Booking_ID"),
    model_udf(*[col(c) for c in columns]).alias("Predicted_BookingPrice")
)

# COMMAND ----------
feature_table_name = f"{project_config.catalog_name}.{project_config.schema_name}.hotel_booking_price_preds"
fe.create_table(
    name=feature_table_name,
    primary_keys=["Booking_ID"],
    df=preds_df,
    description="Hotel Booking Prices predictions feature table"
)

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

# COMMAND ----------
feature_spec_name = f"{project_config.catalog_name}.{project_config.schema_name}.return_hotel_booking_prices"
features = [
            FeatureLookup(
                table_name=feature_table_name,
                lookup_key="Booking_ID",
                feature_names=["Predicted_BookingPrice"],
            )
        ]
fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------
fe.create_online_store(
    name="hotel-booking-price-preds",
    capacity="CU_1"
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
        entity_name=feature_spec_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
]

workspace = WorkspaceClient()
endpoint_name = "hotel-booking-feature-serving"

workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
        tags=[EndpointTag.from_dict({"key": "project_name", "value": "hotel_booking"})]
    )

# COMMAND ----------
# Call the endpoint
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

serving_endpoint = f"{host}/serving-endpoints/hotel-booking-feature-serving/invocations"

response = requests.post(
    serving_endpoint,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["Booking_ID"],
                              "data": [["INN36285"]]}},
)
response.text
