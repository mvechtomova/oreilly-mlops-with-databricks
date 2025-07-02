# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import json
import mlflow
import os

from dotenv import load_dotenv


if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

mlflow.get_tracking_uri()
# COMMAND ----------
experiment = mlflow.set_experiment(experiment_name="/Shared/demo")
mlflow.set_experiment_tags({"repository_name": "mvechtomova/oreilly-mlops-with-databricks"})

print(experiment)
# COMMAND ----------
# dump class attributes in a json file for visualization
if not os.path.exists("../demo_artifacts"):
    os.mkdir("../demo_artifacts")
with open("../demo_artifacts/mlflow_experiment.json", "w") as json_file:
    json.dump(experiment.__dict__, json_file, indent=4)

# COMMAND ----------
# get experiment by id
mlflow.get_experiment(experiment.experiment_id)
# COMMAND ----------
# search for experiment
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='mvechtomova/oreilly-mlops-with-databricks'"
)
print(experiments)

# COMMAND ----------
# start a run
with mlflow.start_run(
   run_name="demo-run",
   tags={"git_sha": "1234567890abcd",
         "branch": "chapter_3"},
   description="demo run",
) as run:
    run_id = run.info.run_id
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# COMMAND ----------
run_info = mlflow.get_run(run_id=run_id)

# COMMAND ----------
run_info_dict = run_info.to_dictionary()
with open("../demo_artifacts/run_info.json", "w") as json_file:
    json.dump(run_info_dict, json_file, indent=4)

# COMMAND ----------
print(run_info_dict["data"]["metrics"])

# COMMAND ----------
print(run_info_dict["data"]["params"])

# COMMAND ----------
# search for runs

from time import time
time_hour_ago = int(time() - 3600) * 1000

runs = mlflow.search_runs(
    search_all_experiments=True, #or experiment_ids=[], or experiment_names=[]
    order_by=["start_time DESC"],
    filter_string="status='FINISHED' AND "
                  f"start_time>{time_hour_ago} AND "
                  "run_name LIKE '%demo-run%' AND "
                  "metrics.metric3>0 AND "
                  "tags.mlflow.source.type!='JOB'"
)

# COMMAND ----------
mlflow.start_run(run_id=run_id)
mlflow.log_metric(key="metric3", value=3.0)
# dynamically log metric (trainings epochs)
for i in range(0,3):
    mlflow.log_metric(key="metric1", value=3.0+i/2, step=i)
mlflow.log_artifact("../demo_artifacts/mlflow_meme.jpg")
mlflow.log_text("hello, MLflow!", "hello.txt")
mlflow.log_dict({"k": "v"}, "dict_example.json")
mlflow.log_artifacts("../demo_artifacts", artifact_path="demo_artifacts")

# COMMAND ----------
# log figure
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot([0, 1], [2, 3])
mlflow.log_figure(fig, "figure.png")

for i in range(0,3):
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mlflow.log_image(image, key="demo_image", step=i)

mlflow.end_run()

# COMMAND ----------
# load objects
artifact_uri = runs.artifact_uri[0]
dict_example = mlflow.artifacts.load_dict(
    f"{artifact_uri}/dict_example.json")
figure = mlflow.artifacts.load_image(f"{artifact_uri}/figure.png")
text = mlflow.artifacts.load_text(f"{artifact_uri}/hello.txt")

# COMMAND ----------
# download artifacts
if not os.path.exists("../downloaded_artifacts"):
    os.mkdir("../downloaded_artifacts")
mlflow.artifacts.download_artifacts(
    artifact_uri=f"{artifact_uri}/demo_artifacts",
    dst_path="../downloaded_artifacts")

# COMMAND ----------
# nested runs: useful for hyperparameter tuning
with mlflow.start_run(run_name="top_level_run") as run:
    for i in range(1,5):
        with mlflow.start_run(run_name=f"subrun_{str(i)}", nested=True) as subrun:
            mlflow.log_metrics({"m1": 5.1+i,
                                "m2": 2*i,
                                "m3": 3+1.5*i})
