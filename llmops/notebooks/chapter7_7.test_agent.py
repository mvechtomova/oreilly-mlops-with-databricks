# Databricks notebook source

import random
from datetime import datetime

import mlflow

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()

cfg = ProjectConfig.from_yaml("../project_config.yml")
mlflow.set_experiment(cfg.experiment_path)

# COMMAND ----------
# Set the agent
agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
    lakebase_project_id=cfg.lakebase_project_id
)

mlflow.models.set_model(agent)

# COMMAND ----------
# First request
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"

first_request = {
    "input": [
        {"role": "user",
         "content": "What are recent papers about LLMs and reasoning?"}
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": f"req-{timestamp}-1",
    },
}

result_1 = agent.predict(first_request)
print(result_1.model_dump(exclude_none=True))

# COMMAND ----------
# Second request — same session, references the first conversation
second_request = {
    "input": [
        {"role": "user",
         "content": "Rewrite that post focusing on chain-of-thought prompting."}
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": f"req-{timestamp}-2",
    },
}

result_2 = agent.predict(second_request)
print(result_2.model_dump(exclude_none=True))


# COMMAND ----------
for chunk in agent.predict_stream(second_request):
   print(chunk.model_dump(exclude_none=True))
