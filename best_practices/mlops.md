# MLOps track: classical ML lifecycle practices

Reference implementation: [mlops/](../mlops/) (hotel booking price prediction,
LightGBM + MLflow + Unity Catalog + Declarative Automation Bundles). Foundations
([foundations.md](foundations.md)) apply on top of everything here.

## 1. Data management and reproducibility

**Practice: every training run can be reproduced from what MLflow recorded.**

- Data lives in Unity Catalog Delta tables with Change Data Feed enabled at creation
  (`delta.enableChangeDataFeed = true`), so downstream consumers can read increments.
- Reads for training are pinned to a Delta version: resolve the current table version,
  then query with `VERSION AS OF <n>`. See
  [mlops/src/hotel_booking/data/data_loader.py](../mlops/src/hotel_booking/data/data_loader.py).
- Train/test split is **time-based**, not random: a trailing training window and a
  held-out most-recent window computed from the data's date column. This prevents
  temporal leakage and mirrors how the model is used.
- Dataset lineage is logged to MLflow: `mlflow.data.from_spark(df, sql=query)` plus
  `mlflow.log_input(..., context="training"/"testing")`, capturing the exact SQL and
  Delta version behind each run.
- Preprocessing is idempotent and re-runnable: it detects whether the target table
  exists and appends new (synthetic, in the course setting) data rather than failing
  or duplicating.

## 2. Training, tracking, and experiment hygiene

**Practice: a model artifact is useless unless its signature, inputs, metrics, and
provenance are logged with it.**

- Experiments have stable, shared names (`/Shared/<project>-training`), not
  per-user defaults.
- Every logged model has an `infer_signature(...)` signature and an `input_example`.
  If the model takes `params` at inference time, the signature declares a params
  schema (otherwise params are silently dropped at serving).
- Hyperparameters and metrics are logged; model quality is computed with a standard
  evaluation (`mlflow.models.evaluate(model_type=...)`) rather than ad-hoc prints.
- Every run and registered version is tagged with provenance: `git_sha`, `branch`,
  and the job `run_id`, passed from CI/CD → bundle variable → job parameter → code.
  See the `Tags` model in
  [mlops/src/hotel_booking/config.py](../mlops/src/hotel_booking/config.py).

## 3. Registry and gated promotion

**Practice: register in Unity Catalog, promote by alias, and only ship improvements.**

- Models register under a three-level UC name (`{catalog}.{schema}.{model}`), so
  environment separation comes from the catalog, not from name suffixes.
- Promotion uses **aliases** (`@latest-model`), never legacy stages. Consumers
  resolve versions by alias.
- **Champion/challenger gate**: before registering, the current champion
  (`@latest-model`) is evaluated on the same held-out data as the challenger; the
  challenger is registered only if it is better (for example lower RMSE). See
  [mlops/scripts/train_register_model.py](../mlops/scripts/train_register_model.py).
- The training task signals downstream via task values
  (`dbutils.jobs.taskValues.set("model_updated", ...)`), and the bundle's
  `condition_task` skips deployment when nothing improved. No-op deploys are avoided
  by design, not by luck.

## 4. Model packaging for serving

**Practice: the served model is self-contained; dependency problems surface at logging
time, not as a serving-time `ModuleNotFoundError`.**

Two acceptable patterns, both demonstrated:

- **Self-contained model**: custom transformers defined where the model can carry
  them (the `CatToIntTransformer`-inside-`train` trick), so serving needs no private
  package.
- **Custom pyfunc with the package attached**: a `mlflow.pyfunc.PythonModel` wrapper
  that loads the underlying pipeline in `load_context` and post-processes
  predictions, logged with `code_paths=[<project wheel>]` and a `conda_env` that
  pip-installs that wheel. The wheel name is derived from `version.txt`, not
  hardcoded. See
  [mlops/src/hotel_booking/models/pyfunc_model_wrapper.py](../mlops/src/hotel_booking/models/pyfunc_model_wrapper.py).

Also graded here: awareness of the serving contract. With feature-engineering
training sets (`fe.create_training_set`), the input dataframe schema becomes the
endpoint's request schema, so unused columns must be dropped before creating the
training set.

## 5. Serving

**Practice: endpoints are code, idempotent, observable, and cost-attributed.**

- Endpoint creation is create-or-update: list existing endpoints, `create` if absent,
  `update_config` otherwise, so re-runs and redeploys are safe. See
  [mlops/src/hotel_booking/models/serving.py](../mlops/src/hotel_booking/models/serving.py).
- Inference payload capture is on from day one (AI Gateway inference table config
  writing request/response to a UC table): monitoring depends on it.
- `scale_to_zero_enabled` and a deliberate `workload_size` are set, plus
  `budget_policy_id` for cost attribution.
- The endpoint serves a version resolved by alias, and traffic-splitting (A/B) routes
  on request `params`, not on fields the scoring server strips.

## 6. Monitoring and alerting

**Practice: close the loop; a model without monitoring is a prototype.**

Reference: [mlops/src/hotel_booking/monitoring.py](../mlops/src/hotel_booking/monitoring.py)
and [mlops/resources/ml_monitoring.yml](../mlops/resources/ml_monitoring.yml),
[mlops/resources/alert.yml](../mlops/resources/alert.yml).

- A scheduled job parses the raw inference payload table into a typed monitoring
  table: explicit request/response `StructType` schemas, `from_json`, explode and
  flatten records.
- Processing is **incremental**: read only payloads newer than the watermark
  (`MAX(request_time)` already in the monitoring table); a no-op run appends nothing
  and exits cleanly.
- Predictions are joined with ground truth as it arrives, keyed on a business id that
  travels with the request (for example `client_request_id`).
- Lakehouse monitoring (data quality / inference profile) is attached to the
  monitoring table idempotently: refresh when the monitor exists, create when it
  does not.
- An alert closes the loop: a scheduled SQL alert on the monitor's metrics table
  (for example "% of rows with MAE above threshold") with a notification
  subscription. The alert and its dedicated warehouse are bundle resources, not
  UI-created.

## 7. The pipeline as one deployable DAG

**Practice: preprocess → train/register → (condition) → deploy is one scheduled job,
defined in the bundle.**

See [mlops/resources/ml_pipeline.yml](../mlops/resources/ml_pipeline.yml):

- Tasks are `spark_python_task`s running the thin `scripts/` entry points from the
  deployed wheel, with `depends_on` expressing order and a `condition_task` gating
  deployment on the training outcome.
- Every task receives `--root_path`, `--env`, `--git_sha`, `--branch`, and job
  runtime values, so the same code runs identically in dev, acc, and prd.
- The schedule is paused outside prd via the bundle variable.
