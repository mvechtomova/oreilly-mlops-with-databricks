# Project gotchas

Non-obvious behaviors worth flagging in the book or remembering across sessions. Add new entries as they come up. Keep each one short: what happens, why it matters, where it bit us.

Overall rules (conventions, testing, dependencies) come first; chapter-specific gotchas follow, sorted by chapter.

## Conventions

### Record session feedback here

Whenever the user gives feedback or states a preference during a session (how to structure things, tools to use, conventions to follow), write it into this file as a rule so it survives across sessions. Do this as part of acting on the feedback, not only when asked.

### Use `uv` for everything, including testing

Run all Python tooling through `uv`: `uv sync --extra dev` (or `--extra ci`) to set up, `uv run pytest` to test, `uv run <tool>` for anything else. Do not invoke `python`, `pip`, or `pytest` directly against the system or a hand-activated venv.

### On any code change, run tests and checks (and keep tests in sync with the SDK)

After every change to `src/` (or anything tests cover), run both before calling it done:

- Tests: `uv run --extra dev pytest tests/` — all must pass.
- Lint + format: `cd mlops && uv run pre-commit run --all-files`. Pre-commit owns ruff check + ruff-format (pinned in `.pre-commit-config.yaml`); it auto-fixes what it can and reports the rest. It must be run from `mlops/` (that is where the config resolves) and end clean. Line length stays within 90 (`[tool.ruff] line-length`).

When the change touches a Databricks/MLflow SDK call, **verify the API against the installed SDK version** (inspect signatures/enums/dataclass fields, e.g. `databricks-sdk` 0.102.0's `data_quality` API) and **update the tests to match** — a renamed param, moved module, or new required field (e.g. `Refresh` requires `object_type`/`object_id`) should be caught by a unit test, not at runtime. If no test covers the changed code, add one (mock `WorkspaceClient`; see [mlops/tests/test_monitoring.py](mlops/tests/test_monitoring.py) and [mlops/tests/test_serving.py](mlops/tests/test_serving.py)).

### Respect the line length

Keep every line within the ruff `line-length` (90) set in [mlops/pyproject.toml](mlops/pyproject.toml). This applies to code, comments, and docstrings. Wrap or rephrase rather than letting a line run over.

### Ruff per-file ignores live in the file, not in `pyproject.toml`

Prefer a file-level `# ruff: noqa: ...` directive over a `[tool.ruff.lint.per-file-ignores]` entry, so the suppression sits next to the code it covers. The directive needs explicit rule codes, not a category prefix: use `# ruff: noqa: ANN001, ANN201, ANN202` (not `# ruff: noqa: ANN`), and keep it on its own line with no trailing comment. Test modules use exactly this to leave fixtures (`mocker`, `cfg`, ...) un-annotated.

Exceptions that get a *blanket* `# ruff: noqa` (no codes) because they are not lint-clean by design: every notebook (line 2, right under `# Databricks notebook source` — book code, not enforced), the Databricks job scripts that reach for the injected `dbutils` global (use `# ruff: noqa: F821` there), and the `mlops/typings/__builtins__.pyi` runtime stub. ruff-format still reformats notebooks, which is fine.

### Always quote bundle variable references in YAML

In Databricks Asset Bundle YAML (`databricks.yml` and everything under `resources/`), always wrap any value that is or contains a `${...}` substitution in double quotes — e.g. `pause_status: "${var.schedule_pause_status}"`, `- "${workspace.root_path}"`, `root_path: "/Shared/.bundle/${bundle.target}/${bundle.name}"`. Unquoted forms also work, but mixing them within a file (the preprocessing task unquoted, the train task quoted) reads as inconsistent in the book. Quote everywhere.

### Attribute every job's cost: usage policy for serverless, tag for classic compute

Cost attribution must reach `system.billing.usage.custom_tags` on every job. Two cases, because the two compete:

- **Serverless jobs** carry `budget_policy_id: "${var.usage_policy_id}"` (a job-level field), NOT a `tags:` block. The usage policy injects its own tags into the billing rows. See [mlops/resources/ml_pipeline.yml](mlops/resources/ml_pipeline.yml). **A usage policy is ALWAYS applied to a serverless resource, even when you do not pass one**: if none is set, Databricks auto-assigns the first policy in alphabetical order, and that policy's tags land on your usage. So tagging a serverless job with `project` alone does NOT work, the (auto-applied) policy's `project` tag still wins on the key collision and your tag is silently dropped. The only reliable control is to assign the right policy explicitly; do not also add a same-key `project` tag to a policy-covered resource (it will be ignored). This is confirmed behavior, easy to miss because the job still "has a tag" in the YAML.
- **Classic-compute jobs** can't take a usage policy, so they keep an explicit `tags:` block with key `project` (e.g. `project: "hotel-booking"`). The only one is the monitoring job in [mlops/resources/ml_monitoring.yml](mlops/resources/ml_monitoring.yml).
- **SQL warehouses** also keep `custom_tags` (key `project`) because serverless SQL warehouses did not support a usage policy at the time of writing. See [mlops/resources/alert.yml](mlops/resources/alert.yml).

`usage_policy_id` is defined once as a bundle variable in each `databricks.yml` (default matches `project_config.yml`), so the policy id lives in one place per project.

### Notebook imports: introduce in the cell where first used

In `mlops/notebooks` (and llmops notebooks), do NOT hoist all imports to the top of the file. Put each import in the `# COMMAND ----------` cell where it is first used, so the book can show the import alongside the code that needs it. Within each cell's import block, sort per PEP / isort rules (standard library, third party, then first party, alphabetical within groups). See [mlops/notebooks/chapter4_1.model_serving.py](mlops/notebooks/chapter4_1.model_serving.py) as the reference example.

## Testing and dependencies

### `pyspark` must come from `databricks-connect`, never installed on its own

`pyspark` is a transitive dependency of `databricks-feature-engineering` (mlops) and of `databricks-agents` (llmops), and `databricks-connect` is what actually provides a working `pyspark` import. Installing a standalone `pyspark` alongside `databricks-connect` does NOT error, but it silently does not work (the two clash).

- Consequence for tests: use `databricks-connect` for testing. The `ci` extra in [mlops/pyproject.toml](mlops/pyproject.toml) therefore includes `databricks-connect` so that modules importing `pyspark`/`delta` (e.g. `data_loader`, `data_processor`, `common`) import cleanly on a GitHub runner. Unit tests mock the Spark session, so no cluster is started.
- Do not add a bare `pyspark` pin to any extra.

## MLOps chapters

### Chapter 3: `CatToIntTransformer` is defined inside `LightGBMModel.train`

LightGBM supports integer-encoded categorical features, which often performs better than one-hot encoding (see the LightGBM docs for details). A custom encoder is needed so the model treats the integer-encoded features as categorical, and so earlier-unseen categories get `-1` assigned to avoid errors while computing predictions.

- The `CatToIntTransformer` class is defined *inside* the `train` method of `LightGBMModel` (in [mlops/src/hotel_booking/models/lightgbm_model.py](mlops/src/hotel_booking/models/lightgbm_model.py)). This isn't ideal from a design standpoint, but it avoids needing the `hotel_booking` package available when serving the model, keeping the model self-contained and easier to deploy.
- Better ways to handle private dependencies are covered later, with custom pyfunc models.

### Chapter 3: custom pyfunc wrapper needs `code_paths` + `conda_env` for module-level helpers

`HotelBookingModelWrapper` (in [mlops/src/hotel_booking/models/pyfunc_model_wrapper.py](mlops/src/hotel_booking/models/pyfunc_model_wrapper.py)) is a `mlflow.pyfunc.PythonModel` that loads the sklearn pipeline in `load_context` and post-processes predictions (adds a 5% commission, rounds to 2 decimals) — the functionality of a FastAPI wrapper, but logged with the model.

- The `adjust_price` helper is defined at *module level*, outside the class. It will NOT be importable at the serving step unless the custom package is logged via `code_paths=[...]` and `conda_env=...` in `log_register_model`. Forgetting this gives a serving-time `ModuleNotFoundError`, not a logging error, so it bites late.
- This is the clean alternative to the self-contained `CatToIntTransformer`-inside-`train` trick above: logging the package with `code_paths` + `conda_env` makes all modules accessible regardless of how the code is structured, so helpers can live wherever is natural.
- `log_register_model` takes the wrapped model's `ModelInfo`, the pyfunc model name to register under, the experiment name, tags, and `code_paths`.

### Chapter 4: `fe.create_training_set` bakes the input df schema into the serving contract

When using `FeatureEngineeringClient.create_training_set(df=train_set, label=..., feature_lookups=[...])` followed by `fe.log_model(training_set=...)`, the served endpoint expects every column of `train_set` at request time (minus the label), even columns the underlying sklearn pipeline never uses.

- Plain `mlflow.sklearn.log_model` + `infer_signature(X_train, ...)` does NOT behave this way; the signature reflects only what was sliced into `X_train`.
- In [mlops/notebooks/chapter4_3.model_serving_feature_lookup.py](mlops/notebooks/chapter4_3.model_serving_feature_lookup.py), this is why `train_set` drops `P_C`, `P_not_C`, `booking_status` (genuinely unused) and `lead_time`, `arrival_month`, `repeated` (re-introduced by lookups/functions) before being passed to `create_training_set`.
- Worth calling out in the book: contrasts cleanly with the non-FE flow shown earlier.

### Chapter 4: serverless resource creation: check-if-exists + pass the usage policy

Creating serverless resources (feature specs, online stores, serving endpoints, ...) raises if the resource already exists, so guard creation (try/except or an existence check) to keep notebooks/jobs idempotent on re-run. See the `create_feature_spec` / `create_online_store` try/except in [mlops/notebooks/chapter4_2.feature_serving.py](mlops/notebooks/chapter4_2.feature_serving.py).

- Online stores must reach the `AVAILABLE` state before you can publish to them, so poll `fe.get_online_store(...)` until ready.
- The budget/usage policy id (`cfg.usage_policy_id`) must be passed to ALL serverless resources, via `budget_policy_id=...` (serving endpoints) or `usage_policy_id=...` (online stores). Omitting it is an easy thing to miss.
- `EndpointCoreConfigInput` requires `name` under databricks-sdk 0.102.0: pass `name=endpoint_name` inside it, e.g. `EndpointCoreConfigInput(name=endpoint_name, served_entities=...)`. Omitting it raises `TypeError: EndpointCoreConfigInput.__init__() missing 1 required positional argument: 'name'`. This bit us in [mlops/src/hotel_booking/models/serving.py](mlops/src/hotel_booking/models/serving.py) and the chapter 4 serving notebooks.

### Chapter 5: override the job cluster with an interactive one in `dev` (classic compute)

When a job uses classic compute, a bundle target can override the job's default (new) job cluster with an existing interactive cluster by setting `cluster_id: <id>` on the target. Jobs then run on that running cluster instead of spinning up a fresh one — commonly done in the `dev` target to speed up iteration and debugging (does not apply to serverless jobs). Pair it with `dynamic_version: true` on the artifact so the latest package is reinstalled whenever the code changes, which matters during active development:

```yaml
targets:
  dev:
    default: true
    mode: development
    cluster_id: 0513-112814-kzixtb9s
    workspace:
      host: <your_host>
      root_path: "/Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.target}/${bundle.name}"
    artifacts:
      default:
        type: whl
        build: uv build
        path: .
        dynamic_version: true
```

The committed [mlops/databricks.yml](mlops/databricks.yml) keeps the `dev` target on default job clusters (no `cluster_id`, no `dynamic_version`); the override above is the local dev-speed tweak described in the chapter.

## LLMOps chapters

### Vector search: endpoint and index creation are async; wait before using them

Creating a vector search endpoint or index returns before the resource is
`ONLINE`, and `get_index` hands back a usable handle even while the index is
still `PROVISIONING`/updating. Operating on a not-yet-ready resource (e.g.
calling `index.sync()`, `similarity_search()`, or the agent querying it right
after deploy) raises "endpoint/index not ready" errors.

- `VectorSearchManager.create_endpoint_if_not_exists` already waits, via
  `create_endpoint_and_wait`. But `create_or_get_index` uses plain
  `create_delta_sync_index` (no wait), so a freshly created index is returned
  before it is ready. See
  [llmops/src/arxiv_curator/vector_search.py](llmops/src/arxiv_curator/vector_search.py).
- Remedy (databricks-vectorsearch 0.66): use
  `create_delta_sync_index_and_wait`, or call `index.wait_until_ready()` before
  `sync()`/queries; `client.wait_for_endpoint(...)` is the endpoint equivalent.
- Related, separate failure: deleting an endpoint does NOT delete its index.
  `index_exists(endpoint_name=...)` then returns `False` (the endpoint is gone),
  the code calls `create_delta_sync_index`, and UC rejects the still-existing
  index name with "UC entity ... already exists". Drop the orphaned index
  (`client.delete_index(index_name=...)`) before re-creating.

### Trace propagation: agent-framework auth vs an explicit SPN are mutually exclusive

To get production traces flowing to an MLflow experiment, there are two paths, and they must not be mixed. The deciding factor is who authenticates to the experiment, NOT whether you call `agents.deploy` or plain model serving. See [llmops/notebooks/chapter9_2.deploy_agent.py](llmops/notebooks/chapter9_2.deploy_agent.py).

- **Agent-framework auth (default):** call `agents.deploy(...)`. It already authenticates to the MLflow experiment for you (and enables inference tables, creates the UC trace table, and creates the experiment if `MLFLOW_EXPERIMENT_ID` is omitted). It also provisions automatic auth for exactly the resources the agent declared at log/register time via the `resources=[...]` list (the `DatabricksServingEndpoint`, `DatabricksVectorSearchIndex`, `DatabricksGenieSpace`, `DatabricksSQLWarehouse`, `DatabricksTable`, ... in [llmops/notebooks/chapter8_4.evaluate_log_register_agent.py](llmops/notebooks/chapter8_4.evaluate_log_register_agent.py)). This is the auth that breaks: if you set `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` / `DATABRICKS_HOST` in `environment_vars`, they override that framework-managed resource auth, and the agent can no longer reach the resources from its `resources=[...]` list (vector search, Lakebase, Genie, etc.). So in this mode you MUST NOT set those three. Pass only the app-level vars (`GIT_SHA`, `MODEL_VERSION`, `MODEL_SERVING_ENDPOINT_NAME`, `MLFLOW_EXPERIMENT_ID`, and the `LAKEBASE_SP_*` your code reads).
- **Explicit-SPN auth:** create a service principal, grant it `CAN_EDIT` on the experiment, and supply it via `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` / `DATABRICKS_HOST` plus `ENABLE_MLFLOW_TRACING=true` and `MLFLOW_EXPERIMENT_ID`. Once you take on the SPN yourself, the deployment command (`agents.deploy` or `model_serving`) no longer matters, both work. This is the only way to propagate traces when NOT using the agent framework.

So: either let the framework own auth and never set `DATABRICKS_CLIENT_ID`, or own the SPN yourself and set it everywhere. Half-and-half breaks resource auth.

Two more sharp edges with `agents.deploy`:

- **UC-backed experiments do not work for traces yet.** `agents.deploy` provisions a UC trace table, but if the MLflow experiment is created with UC as its backend, trace propagation does not work at the time of writing. Use a classic (non-UC) experiment so traces land where the auth above makes them work.
- **Secrets are resolved at deploy time, not pulled live.** Secret references in `environment_vars` (e.g. `f"{{secrets/{secret_scope}/client-secret}}"`) are evaluated once, at the moment of deployment. The running agent does not re-read the scope. So if you rotate a secret and invalidate the old value, the live agent keeps using the stale value and breaks. To rotate safely: write the new secret, REDEPLOY the agent (so it picks up the new value) while the old value is still valid, and only then invalidate the old one.
