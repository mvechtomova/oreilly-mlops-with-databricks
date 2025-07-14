import os

import mlflow
from delta.tables import DeltaTable
from dotenv import load_dotenv
from pyspark.sql import SparkSession


def set_mlflow_tracking_uri() -> None:
    """
    Set the MLflow tracking URI based on the provided profile.

    """
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        load_dotenv()
        profile = os.environ["PROFILE"]
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")

def get_delta_table_version(
    spark: SparkSession, full_table_name: str
) -> str:
    """
    Get the latest version of a Delta table.

    :param spark: Spark session.
    :param full_table_name: Name of the delta table as {catalog}.{schema}.{name}.
    :return: Latest version of the Delta table.
    """
    delta_table = DeltaTable.forName(spark, full_table_name)
    return str(delta_table.history().select("version").first()[0])
