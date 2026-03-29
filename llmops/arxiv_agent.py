import mlflow
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent

config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "arxiv",
        "genie_space_id": "01f0e97a42981382b3d16f3f1899fdb5",
        "system_prompt": "prompt placeholder",
        "llm_endpoint": "databricks-gpt-oss-120b",
        "lakebase_project_id": "arxiv-agent-lakebase",
    }
)

agent = ArxivAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
    lakebase_project_id=config.get("lakebase_project_id"),
)
mlflow.models.set_model(agent)
