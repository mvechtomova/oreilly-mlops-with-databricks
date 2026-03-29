# Databricks notebook source
import random
from datetime import datetime
from databricks.sdk import WorkspaceClient
from openai import OpenAI


workspace = WorkspaceClient()
host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

endpoint_name = "arxiv-agent-endpoint"

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content": "What are recent papers about LLMs and reasoning?"}
    ],
    extra_body={"custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    }}
)

print(response)

# COMMAND ----------
