# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from hotel_booking.config import ProjectConfig

# COMMAND ----------
cfg = ProjectConfig.from_yaml(
    config_path="../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
inf_table = spark.table(
    f"{catalog}.{schema}.hotel_booking_monitoring_payload")


# inf_table = spark.sql(f"""SELECT FROM {catalog}.{schema}.hotel_booking_monitoring_payload
#     WHERE request_time > (
#         SELECT MAX(request_time) AS TIMESTAMP
#         FROM {catalog}.{schema}.model_monitoring)
# """)



# COMMAND ----------
request_schema = StructType([
    StructField("dataframe_records", ArrayType(StructType([
        StructField("number_of_adults", IntegerType(), True),
        StructField("number_of_children", IntegerType(), True),
        StructField("number_of_weekend_nights", IntegerType(), True),
        StructField("number_of_week_nights", IntegerType(), True),
        StructField("car_parking_space", IntegerType(), True),
        StructField("special_requests", IntegerType(), True),
        StructField("lead_time", IntegerType(), True),
        StructField("type_of_meal", StringType(), True),
        StructField("room_type", StringType(), True),
        StructField("arrival_month", IntegerType(), True),
        StructField("market_segment_type", StringType(), True),
    ])), True)
])


response_schema = StructType([
    StructField("predictions", StructType([
        StructField("Total price per night",
                    ArrayType(DoubleType()), True)
    ]), True)
])

# COMMAND ----------
# parse and explode
inf_table_parsed = (
    inf_table
    .withColumn("p_request",
                F.from_json(F.col("request"),
                            request_schema))
    .withColumn("p_response",
                F.from_json(F.col("response"),
                            response_schema))
    .withColumn("prediction",
                F.col("p_response.predictions.`Total price per night`")[0])
    .withColumn("record",
                F.explode(F.col("p_request.dataframe_records")))
)
# COMMAND ----------

df_final = inf_table_parsed.select(
    "request_time",
    "databricks_request_id",
    "execution_duration_ms",
    F.col("client_request_id").alias("Booking_ID"),
    F.col("record.number_of_adults").alias("number_of_adults"),
    F.col("record.number_of_children").alias("number_of_children"),
    F.col("record.number_of_weekend_nights").alias("number_of_weekend_nights"),
    F.col("record.number_of_week_nights").alias("number_of_week_nights"),
    F.col("record.car_parking_space").alias("car_parking_space"),
    F.col("record.special_requests").alias("special_requests"),
    F.col("record.lead_time").alias("lead_time"),
    F.col("record.type_of_meal").alias("type_of_meal"),
    F.col("record.room_type").alias("room_type"),
    F.col("record.arrival_month").alias("arrival_month"),
    F.col("record.market_segment_type").alias("market_segment_type"),
    "prediction",
    F.lit("hotel-booking-pyfunc").alias("model_name")
)

# COMMAND ----------
ground_truth_set = spark.table(f"{catalog}.{schema}.hotel_booking")

df_final_with_status = df_final.join(
    ground_truth_set.select("Booking_ID", "average_price"),
    on="Booking_ID", how="left")
# COMMAND ----------
monitoring_table = f"{catalog}.{schema}.model_monitoring"

df_final_with_status.write.format(
    "delta").mode("append").saveAsTable(monitoring_table)

# Important to update monitoring
spark.sql(f"""ALTER TABLE {monitoring_table}
           SET TBLPROPERTIES (delta.enableChangeDataFeed = true);""")

# COMMAND ----------
# create monitor

w = WorkspaceClient()

w.quality_monitors.create(
    table_name=monitoring_table,
    assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
    output_schema_name=f"{catalog}.{schema}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
        prediction_col="prediction",
        timestamp_col="request_time",
        granularities=["5 minutes"],
        model_id_col="model_name",
        label_col="average_price",
    ),
)

# try:
#     w.quality_monitors.get(monitoring_table)
#     w.quality_monitors.run_refresh(table_name=monitoring_table)
#     logger.info("Lakehouse monitoring exist, refreshing. monitor")
# except NotFound:
#     logger.info("Creating monitor.")

# hotel_booking
# └── Tables
#     ├── hotel_booking_monitoring_payload
#     ├── model_monitoring
#     ├── model_monitoring_drift_metrics
#     └── model_monitoring_profile_metrics
