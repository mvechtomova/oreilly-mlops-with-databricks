import os

import mlflow
from delta.tables import DeltaTable
from dotenv import load_dotenv
from pyspark.sql import SparkSession
import argparse
from collections.abc import Sequence
from databricks.sdk import WorkspaceClient


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

def create_parser(args: Sequence[str] = None) -> argparse.Namespace:
    """Create and configure an argument parser for MLOps on Databricks.

    This function sets up a parser with subparsers for different MLOps operations.

    :param args: Optional sequence of command-line argument strings
    :return: Parsed argument namespace
    """
    parser = argparse.ArgumentParser(description="Parser for MLOps on Databricks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--root_path", type=str, required=True, help="Path of root on DAB")
    common_args.add_argument("--env", type=str, required=True, help="Path of env file on DAB")

    # Data ingestion subparser
    subparsers.add_parser("common", parents=[common_args], help="Common processor")

    # Model training & registering subparser
    model_parser = subparsers.add_parser(
        "train_deploy_model",
        parents=[common_args],
        help="Model training and registering",
    )
    model_parser.add_argument("--branch", type=str, required=True, help="branch of the project")
    model_parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
    model_parser.add_argument("--run_id", type=str, required=True, help="run id of the run of the Lakeflow job")
    model_parser.add_argument("--job_id", type=str, required=True, help="Lakeflow job id")
 
    approve_parser = subparsers.add_parser("approve_check", parents=[common_args], help="Approval check for the model")
    approve_parser.add_argument("task_name", type=str, help="Name of the task to approve")

    return parser.parse_args(args)
