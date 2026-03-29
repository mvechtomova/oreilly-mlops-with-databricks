# Databricks notebook source
import mlflow
from arxiv_curator.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()

# COMMAND ----------
mlflow.set_experiment('/Shared/tracing-demo')

@mlflow.trace
def my_func(x, y):
    return x + y

# COMMAND ----------
def my_func(x, y):
    return x + y

with mlflow.start_span("my_function") as span:
    x = 1
    y = 2
    span.set_inputs({"x": x, "y": y})
    result = my_func(x, y)
    span.set_outputs({"output": result})


# COMMAND ----------
import random
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

git_sha = "abcd"

@mlflow.trace
def my_func(x, y):
    mlflow.update_current_trace(
    metadata={
        "mlflow.trace.session": session_id,
    },
    tags={"model_serving_endpoint_name": "arxiv-agent-endpoint",
          "model_version": 1,
          "git_sha": git_sha},
    client_request_id=request_id
)
    return x + y

# COMMAND ----------
# search traces

mlflow.search_traces(filter_string=f"tags.git_sha = '{git_sha}'")
