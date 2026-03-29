# Databricks notebook source
from databricks.sdk import WorkspaceClient
from openai import OpenAI

# Authenticate using Databricks SDK
w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

# Configure OpenAI client to use Databricks serving endpoint
client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints"
)

# Send a random test question
response = client.chat.completions.create(
    model="databricks-gpt-oss-120b",
    messages=[
        {"role": "user",
         "content": "What is MLOps?"}
    ]
)

# Print the response
print("Response from Databricks serving endpoint:")
print(response.choices[0].message.content)

# COMMAND ----------
