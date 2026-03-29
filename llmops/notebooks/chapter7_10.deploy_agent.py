# Databricks notebook source
from databricks import agents
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

from arxiv_curator.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = "arxiv-agent-endpoint"
secret_scope = "arxiv-agent-scope"

model_version = MlflowClient().get_model_version_by_alias(
    model_name, "latest-model").version

workspace = WorkspaceClient()
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_path)

# COMMAND ----------
git_sha = "local"

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client-id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client-secret}}}}",
        "LAKEBASE_SP_HOST": workspace.config.host,
    },
)
