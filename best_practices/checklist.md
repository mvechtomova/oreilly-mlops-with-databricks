# Repository evaluation checklist

Use this to evaluate any student repository against the practices in
[foundations.md](foundations.md), [mlops.md](mlops.md), and [llmops.md](llmops.md).
Tool substitutions are fine (poetry for uv, GitLab CI for GitHub Actions, plain
`logging` for loguru): grade the practice, not the tool.

## Scoring

- Score each **area** 0, 1, or 2:
  - **2**: all core items present and working.
  - **1**: partially present (some core items missing or done superficially).
  - **0**: absent.
- **[core]** items count toward the area score. **[advanced]** items do not lower a
  score but justify calling an area "excellent"; note them separately.
- Always record **evidence**: the file path (and line if useful) that satisfies or
  fails each item. "No evidence found after checking X, Y, Z" is also evidence.
- Foundations areas apply to every repo. Apply MLOps areas, LLMOps areas, or both
  depending on the project's track.

Suggested report format per area: `score | one-line verdict | evidence | top fix`.

---

## Foundations (all repos)

### F1. Repository and package layout

- [ ] [core] Reusable logic lives in an installable package (`src/<package>/` layout
      preferred), named after the use case. *Verify: `pyproject.toml` packages config;
      `src/` exists.*
- [ ] [core] Notebooks contain no logic the pipeline depends on; they import the
      package. *Verify: grep notebooks for class/function definitions that jobs need.*
- [ ] [core] Thin entry points (scripts or deployment notebooks) separate from both
      package and demo notebooks. *Verify: `scripts/` or
      `resources/deployment_notebooks/` exist and only parse args + call the package.*
- [ ] [core] Tests live in `tests/` and mirror package modules.
- [ ] [core] README explains the structure and how to run things.
- [ ] [advanced] Empty `__init__.py` files; no import-time side effects.

### F2. Dependencies and environment

- [ ] [core] `pyproject.toml` plus a committed lockfile (`uv.lock`, `poetry.lock`).
- [ ] [core] Runtime dependencies pinned exactly; tooling may use ranges.
- [ ] [core] Python pinned to a minor version consistent with the Databricks runtime
      or serverless environment version used in the bundle.
- [ ] [core] Separate `dev` / `ci` extras (or dependency groups).
- [ ] [core] No standalone `pyspark` dependency (must come via `databricks-connect`).
      *Verify: grep pyproject for a bare `pyspark` pin.*
- [ ] [advanced] Version single-sourced (`version.txt` + `dynamic = ["version"]`).

### F3. Configuration management

- [ ] [core] One config file with per-environment blocks (`dev`/`acc`/`prd` or
      similar); shared keys not duplicated.
- [ ] [core] Config loaded through a validated typed model (pydantic or equivalent)
      that rejects unknown environments.
- [ ] [core] No hardcoded catalog/schema/workspace-URL/endpoint names in `src/`,
      entry points, or job YAML. *Verify: grep for literal catalog names and
      `https://.*databricks` outside config and databricks.yml.*
- [ ] [core] Environment flows from deployment (`${bundle.target}` → job param /
      widget → config loader).
- [ ] [core] No secrets in code, config files, or git history. *Verify: grep for
      tokens, client secrets, `dapi`, passwords; check `.env` is gitignored.*
- [ ] [advanced] Validation bounds on hyperparameters / critical values.

### F4. Code quality

- [ ] [core] Pre-commit config with pinned versions: hygiene hooks + lint + format.
- [ ] [core] Linter configured with explicit rules and line length in
      `pyproject.toml`; repo is actually clean. *Verify: run the linter.*
- [ ] [core] Type annotations in package code.
- [ ] [core] Structured logging in library code, not `print`.
- [ ] [advanced] Custom domain exceptions; local, code-specific lint suppressions
      only.

### F5. Testing

- [ ] [core] Unit tests exist for package logic (config, transformations, model
      helpers) and pass offline: no cluster, no network. *Verify: run the suite in an
      isolated env.*
- [ ] [core] External boundaries (Spark, `WorkspaceClient`, mlflow, vector search)
      are mocked; the project's own logic is not mocked away.
- [ ] [core] Mocks are signature-bound: `autospec=True`, or real SDK
      dataclasses/enums where autospec cannot apply. *Verify: grep tests for
      `autospec`.*
- [ ] [core] Idempotent code paths tested on both branches (create vs update/refresh).
- [ ] [advanced] Testing strategy documented (for example `tests/README.md`);
      edge-case fixtures (invalid dates, empty payloads); coverage measured.

### F6. CI/CD

- [ ] [core] CI on pull requests: dependency install from lockfile, lint/format
      check, unit tests. All green on main.
- [ ] [core] CD on merge: bundle deploy to shared environments, run by CI with a
      service principal, not by hand from a laptop.
- [ ] [core] No long-lived credentials in the pipeline: OIDC / secret manager /
      GitHub Environments; no PATs in repo secrets.
- [ ] [core] `git_sha` (and ideally branch) passed into the deploy as a bundle
      variable.
- [ ] [advanced] Actions pinned by commit SHA; per-project path filters that actually
      match (`folder/**`, not `folder`); matrix deploy over acc/prd with environment
      protection rules.

### F7. Declarative Automation Bundles

- [ ] [core] `databricks.yml` with `dev` (default, `mode: development`, per-user
      root path), plus `acc`/`prd` (or at least a prod) targets.
- [ ] [core] All jobs, endpoints-as-jobs, alerts, warehouses, dashboards declared
      under `resources/`; nothing production-critical exists only in the UI.
- [ ] [core] The package wheel is built by the bundle (`artifacts`) and jobs depend
      on it (`environments`/`libraries` with `../dist/*.whl`).
- [ ] [core] Schedules `PAUSED` outside prd via a variable overridden per target.
- [ ] [core] Jobs pass `--env`/`--root_path`/`--git_sha`-style context into tasks.
- [ ] [advanced] All `${...}` references quoted; `condition_task`/`run_job_task` used
      for gating and orchestration; one resource file per concern.

### F8. Cost attribution

- [ ] [core] Serverless jobs/endpoints carry `budget_policy_id` (usage policy);
      classic clusters carry a `project` tag; warehouses carry `custom_tags`.
      *Verify: every resource YAML has one of these.*
- [ ] [core] The policy id / tag key is defined once (bundle variable + config), not
      scattered.
- [ ] [advanced] A cost-monitoring query/notebook/dashboard over
      `system.billing.usage`.

---

## MLOps track

### M1. Data reproducibility

- [ ] [core] Data in UC Delta tables; training reads pinned to a version
      (`VERSION AS OF` or timestamp) or otherwise reproducible.
- [ ] [core] Time-based train/test split where data is temporal (no random split
      over time-dependent data).
- [ ] [core] Dataset lineage logged to MLflow (`mlflow.log_input` with the query).
- [ ] [advanced] Change Data Feed enabled; preprocessing idempotent on re-run.

### M2. Training and tracking

- [ ] [core] Shared experiment names; runs carry params, metrics, and provenance
      tags (`git_sha`, `branch`, job run id).
- [ ] [core] Logged models have a signature and `input_example`; params schema
      declared if `predict` takes params.
- [ ] [core] Evaluation via a standard harness (`mlflow.models.evaluate` or
      equivalent), metrics stored on the run.
- [ ] [advanced] Hyperparameter tuning tracked as child runs / with a tuning
      framework.

### M3. Registry and promotion

- [ ] [core] Models registered in Unity Catalog (`catalog.schema.model`); versions
      resolved by **alias**, not stage or hardcoded version numbers.
- [ ] [core] A promotion gate: challenger compared against the current champion on
      held-out data; registration/deployment only on improvement.
- [ ] [core] The pipeline propagates the decision (task values + `condition_task` or
      equivalent), so deploys are skipped when nothing improved.

### M4. Model packaging

- [ ] [core] The served model is self-contained: either no private imports, or the
      project wheel attached via `code_paths` + environment spec. *Verify: does
      serving need a module that isn't logged with the model?*
- [ ] [core] Any pre/post-processing that belongs to the model travels with it
      (custom pyfunc wrapper), not in the client.
- [ ] [advanced] Wheel name derived from the single-sourced version; awareness of
      the feature-lookup serving contract (unused columns dropped before
      `create_training_set`).

### M5. Serving

- [ ] [core] Endpoint creation is idempotent (create-or-update), runnable repeatedly.
- [ ] [core] Inference payload capture enabled (inference tables / AI gateway).
- [ ] [core] `scale_to_zero`, deliberate workload size, and cost attribution set.
- [ ] [advanced] A/B or traffic-split routing done via request `params` with a
      declared params schema.

### M6. Monitoring and alerting

- [ ] [core] A scheduled job transforms raw inference payloads into a typed
      monitoring table (explicit schemas, not schema inference on JSON strings).
- [ ] [core] Incremental processing with a watermark; empty batches are a clean
      no-op.
- [ ] [core] Ground truth joined in when available, keyed on a business id that
      travels with the request.
- [ ] [core] Lakehouse monitoring (or equivalent drift/quality monitor) attached
      idempotently (create vs refresh).
- [ ] [core] At least one alert on a monitored metric with a notification channel,
      declared as code.

### M7. Pipeline DAG

- [ ] [core] Preprocess → train/register → deploy as one scheduled bundle job with
      explicit `depends_on`.
- [ ] [core] Same code and config path across dev/acc/prd; only the target differs.

---

## LLMOps track

### L1. Agent architecture

- [ ] [core] Agent class lives in the package and implements a standard contract
      (`ResponsesAgent`/`ChatAgent`/pyfunc), with streaming support where relevant.
- [ ] [core] Logged via models-from-code (entry file + `mlflow.models.set_model`),
      with config injected through `model_config`, not hardcoded.
- [ ] [core] System prompt versioned in config (or a prompt registry), not inline in
      the class.

### L2. Robustness

- [ ] [core] Tool loop bounded (`max_iter`) with a graceful cutoff message.
- [ ] [core] Unknown/failed tool calls surface as messages to the model, not
      unhandled exceptions.
- [ ] [core] Retry/backoff on rate limits.
- [ ] [advanced] Optional subsystems (memory, personalization) degrade gracefully;
      connection pools recover from expired short-lived credentials.

### L3. Tracing

- [ ] [core] Typed spans over the whole request path (agent / chain / tool /
      retriever / LLM), token usage captured.
- [ ] [core] Traces tagged with `git_sha`, model version, endpoint name from
      deploy-time env vars.
- [ ] [core] Session and request correlation via `custom_inputs` →
      trace metadata.

### L4. Evaluation

- [ ] [core] Eval dataset versioned in the repo.
- [ ] [core] Mixed scorers: deterministic checks plus LLM judges with explicit
      guidelines.
- [ ] [core] Evaluation runs before registration and its metrics are logged to the
      model's run (a gate, not an afterthought).

### L5. Registration and deployment

- [ ] [core] All touched Databricks resources declared in `resources=[...]` at log
      time.
- [ ] [core] UC registration with provenance tags; promotion and deployment resolve
      by alias.
- [ ] [core] `environment_vars` contain only app-level values and
      `{{secrets/...}}` references; auth mode is consistently framework-managed OR
      explicit SPN, never both. *Verify: if `agents.deploy` is used, no
      `DATABRICKS_CLIENT_ID/SECRET/HOST` in env vars.*
- [ ] [advanced] SPN created and granted as code with least privilege;
      rotation-safe secret handling (rotate → redeploy → invalidate).

### L6. Retrieval infrastructure

- [ ] [core] Vector search endpoint/index creation idempotent (existence checks) and
      readiness-aware (waits before sync/query).
- [ ] [core] Document processing is package code with its own scheduled job; index
      synced incrementally (Delta Sync + CDF).

### L7. Production monitoring and cost

- [ ] [core] Scheduled job evaluates production traces: only new/un-assessed traces,
      cheap scorers on all, LLM judges on a sample; results written back as feedback.
- [ ] [core] An aggregated metrics view (latency, tool calls, tokens, scores) and a
      dashboard, declared as bundle resources.
- [ ] [advanced] GenAI cost breakdown from `system.billing.usage` by SKU and policy.

### L8. Integration testing

- [ ] [core] An end-to-end integration test exists as a bundle job chaining the
      **real** production jobs (`run_job_task`), parameterized with a test flag, and
      is part of the release process.
