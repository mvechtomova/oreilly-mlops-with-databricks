"""Monitoring module for incremental updates of model monitoring table."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from delta.tables import DeltaTable
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
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


class MonitoringManager:
    """Handles incremental updates to the model monitoring table."""

    def __init__(
        self,
        spark: SparkSession,
        config: ProjectConfig,
        payload_table: str = "hotel_booking_monitoring_payload",
        monitoring_table: str = "model_monitoring",
        ground_truth_table: str = "hotel_booking",
        workspace_client: WorkspaceClient | None = None,
    ):
        """Initialize the monitoring table updater.

        :param spark: SparkSession instance
        :param config: ProjectConfig instance with catalog and schema
        :param payload_table: Name of the payload table with inference logs
        :param monitoring_table: Name of the monitoring table to update
        :param ground_truth_table: Name of the ground truth table
        :param workspace_client: Databricks WorkspaceClient instance
        """
        self.spark = spark
        self.config = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.payload_table = f"{self.catalog}.{self.schema}.{payload_table}"
        self.monitoring_table_name = monitoring_table
        self.monitoring_table = (
            f"{self.catalog}.{self.schema}.{monitoring_table}"
        )
        self.ground_truth_table = (
            f"{self.catalog}.{self.schema}.{ground_truth_table}"
        )
        self.w = workspace_client or WorkspaceClient()

        self.request_schema = self._get_request_schema()
        self.response_schema = self._get_response_schema()

    @staticmethod
    def _get_request_schema() -> StructType:
        """Define the schema for request data."""
        return StructType(
            [
                StructField(
                    "dataframe_records",
                    ArrayType(
                        StructType(
                            [
                                StructField(
                                    "number_of_adults", IntegerType(), True
                                ),
                                StructField(
                                    "number_of_children", IntegerType(), True
                                ),
                                StructField(
                                    "number_of_weekend_nights",
                                    IntegerType(),
                                    True,
                                ),
                                StructField(
                                    "number_of_week_nights",
                                    IntegerType(),
                                    True,
                                ),
                                StructField(
                                    "car_parking_space", IntegerType(), True
                                ),
                                StructField(
                                    "special_requests", IntegerType(), True
                                ),
                                StructField("lead_time", IntegerType(), True),
                                StructField(
                                    "type_of_meal", StringType(), True
                                ),
                                StructField("room_type", StringType(), True),
                                StructField(
                                    "arrival_month", IntegerType(), True
                                ),
                                StructField(
                                    "market_segment_type", StringType(), True
                                ),
                            ]
                        )
                    ),
                    True,
                )
            ]
        )

    @staticmethod
    def _get_response_schema() -> StructType:
        """Define the schema for response data."""
        return StructType(
            [
                StructField(
                    "predictions",
                    StructType(
                        [
                            StructField(
                                "Total price per night",
                                ArrayType(DoubleType()),
                                True,
                            )
                        ]
                    ),
                    True,
                )
            ]
        )

    def _get_last_processed_timestamp(self) -> str:
        """Get the last processed timestamp from the monitoring table.

        :return: Last processed timestamp or default value if table is empty
        """
        if self.spark.catalog.tableExists(self.monitoring_table):
            result = self.spark.sql(
                f"""
                SELECT COALESCE(
                    MAX(request_time),
                    CAST('1900-01-01' AS TIMESTAMP)
                ) AS last_time
                FROM {self.monitoring_table}
            """
            ).collect()[0]["last_time"]
            logger.info(f"Last processed timestamp: {result}")
            return result
        else:
            logger.info(
                "Monitoring table does not exist. Processing all records."
            )
            return "1900-01-01"

    def _load_new_payload_data(self) -> DataFrame:
        """Load only new inference logs since the last update.

        :return: DataFrame with new payload data
        """
        last_timestamp = self._get_last_processed_timestamp()

        logger.info(f"Loading payload data from {self.payload_table}")
        inf_table = self.spark.sql(
            f"""
            SELECT * FROM {self.payload_table}
            WHERE request_time > CAST('{last_timestamp}' AS TIMESTAMP)
        """
        )

        record_count = inf_table.count()
        logger.info(f"Found {record_count} new records to process")

        return inf_table

    def _parse_and_transform_payload(
        self, inf_table: DataFrame
    ) -> DataFrame:
        """Parse JSON fields and transform payload data.

        :param inf_table: Raw payload DataFrame
        :return: Transformed DataFrame
        """
        logger.info("Parsing and transforming payload data")

        inf_table_parsed = (
            inf_table.withColumn(
                "p_request",
                F.from_json(F.col("request"), self.request_schema),
            )
            .withColumn(
                "p_response",
                F.from_json(F.col("response"), self.response_schema),
            )
            .withColumn(
                "prediction",
                F.col("p_response.predictions.`Total price per night`")[0],
            )
            .withColumn(
                "record", F.explode(F.col("p_request.dataframe_records"))
            )
        )

        df_final = inf_table_parsed.select(
            "request_time",
            "databricks_request_id",
            "execution_duration_ms",
            F.col("client_request_id").alias("Booking_ID"),
            F.col("record.number_of_adults").alias("number_of_adults"),
            F.col("record.number_of_children").alias("number_of_children"),
            F.col("record.number_of_weekend_nights").alias(
                "number_of_weekend_nights"
            ),
            F.col("record.number_of_week_nights").alias(
                "number_of_week_nights"
            ),
            F.col("record.car_parking_space").alias("car_parking_space"),
            F.col("record.special_requests").alias("special_requests"),
            F.col("record.lead_time").alias("lead_time"),
            F.col("record.type_of_meal").alias("type_of_meal"),
            F.col("record.room_type").alias("room_type"),
            F.col("record.arrival_month").alias("arrival_month"),
            F.col("record.market_segment_type").alias("market_segment_type"),
            "prediction",
            F.lit("hotel-booking-pyfunc").alias("model_name"),
        )

        return df_final

    def _join_with_ground_truth(self, df: DataFrame) -> DataFrame:
        """Join predictions with ground truth labels.

        :param df: DataFrame with predictions
        :return: DataFrame with ground truth labels added
        """
        logger.info("Joining with ground truth data")

        ground_truth_set = self.spark.table(self.ground_truth_table)

        df_with_labels = df.join(
            ground_truth_set.select("Booking_ID", "average_price"),
            on="Booking_ID",
            how="left",
        )

        return df_with_labels

    def _write_to_monitoring_table(
        self, df: DataFrame, use_merge: bool = False
    ) -> None:
        """Write data to the monitoring table.

        :param df: DataFrame to write
        :param use_merge: If True, use MERGE to avoid duplicates.
                         If False, use append mode.
        """
        if df.isEmpty():
            logger.info("No new data to write to monitoring table")
            return

        if not self.spark.catalog.tableExists(self.monitoring_table):
            logger.info(f"Creating monitoring table: {self.monitoring_table}")
            df.write.format("delta").saveAsTable(self.monitoring_table)

            self.spark.sql(
                f"""
                ALTER TABLE {self.monitoring_table}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
            """
            )
            logger.info("Enabled Change Data Feed on monitoring table")

        elif use_merge:
            logger.info("Using MERGE to update monitoring table")
            delta_table = DeltaTable.forName(self.spark, self.monitoring_table)
            (
                delta_table.alias("target")
                .merge(
                    df.alias("source"),
                    "target.databricks_request_id = "
                    "source.databricks_request_id",
                )
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info("MERGE completed successfully")

        else:
            logger.info("Appending data to monitoring table")
            df.write.format("delta").mode("append").saveAsTable(
                self.monitoring_table
            )
            logger.info("Append completed successfully")

    def update_monitoring_table(self, use_merge: bool = False) -> int:
        """Run the complete incremental update process.

        :param use_merge: If True, use MERGE to avoid duplicates.
                         If False, use append mode.
        :return: Number of records processed
        """
        logger.info("Starting incremental monitoring table update")

        inf_table = self._load_new_payload_data()

        if inf_table.isEmpty():
            logger.info("No new data to process")
            return 0

        df_transformed = self._parse_and_transform_payload(inf_table)
        df_final = self._join_with_ground_truth(df_transformed)

        self._write_to_monitoring_table(df_final, use_merge=use_merge)

        record_count = df_final.count()
        logger.info(f"Successfully processed {record_count} records")

        return record_count

    def refresh_monitor(
        self,
        assets_dir: str | None = None,
        prediction_col: str = "prediction",
        timestamp_col: str = "request_time",
        model_id_col: str = "model_name",
        label_col: str = "average_price",
        granularities: list[str] | None = None,
    ) -> None:
        """Create or update a Lakehouse Monitor.

        :param assets_dir: Directory to store monitor assets
        :param prediction_col: Name of the prediction column
        :param timestamp_col: Name of the timestamp column
        :param model_id_col: Name of the model ID column
        :param label_col: Name of the label column
        :param granularities: List of granularities for metrics
        """
        assets_dir = (
            assets_dir
            or f"/Workspace/Shared/lakehouse_monitoring/{self.monitoring_table}"
        )
        granularities = granularities or ["5 minutes"]

        try:
            self.w.quality_monitors.get(self.monitoring_table)
            logger.info(f"Monitor already exists for {self.monitoring_table}")
            self.w.quality_monitors.run_refresh(table_name=self.monitoring_table)
            logger.info(f"Monitor refresh triggered for {self.monitoring_table}")

        except NotFound:
            logger.info(f"Creating new monitor for {self.monitoring_table}")

            self.w.quality_monitors.create(
                table_name=self.monitoring_table,
                assets_dir=assets_dir,
                output_schema_name=f"{self.catalog}.{self.schema}",
                inference_log=MonitorInferenceLog(
                    problem_type=(
                        MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION
                    ),
                    prediction_col=prediction_col,
                    timestamp_col=timestamp_col,
                    granularities=granularities,
                    model_id_col=model_id_col,
                    label_col=label_col,
                ),
            )
            logger.info(f"Monitor created successfully for {self.monitoring_table}")
