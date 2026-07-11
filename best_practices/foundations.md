# Foundations: practices shared by both tracks

These apply to every project, MLOps or LLMOps. Reference implementations:
[mlops/](../mlops/) (hotel booking) and [llmops/](../llmops/) (arxiv curator).

## 1. Repository and package layout

**Practice: ship a real Python package, keep notebooks at the edge.**

- Use a `src/` layout: all reusable logic lives in `src/<package_name>/`, declared via
  `[tool.setuptools.packages.find] where = ["src"]` in `pyproject.toml`. The package is
  named after the use case (`hotel_booking`, `arxiv_curator`), not `src` or `utils`.
- Four clearly separated zones:
  - `src/<package>/`: importable, tested, packaged logic (config, data, models,
    monitoring, utilities).
  - `notebooks/`: exploration and demos. Notebooks import the package; they never
    contain logic the pipeline depends on.
  - `scripts/` or `resources/deployment_notebooks/`: thin entry points that jobs run.
    They parse arguments or widgets, load config, and call into the package. See
    [mlops/scripts/train_register_model.py](../mlops/scripts/train_register_model.py).
  - `tests/`: unit tests mirroring the package modules.
- Deployment definitions (jobs, alerts, dashboards) live in `resources/`, one YAML file
  per concern (pipeline, monitoring, alerting, testing).
- `__init__.py` files stay empty: no import-time side effects.
- The README maps the structure and, for coursework, maps chapters or milestones to
  folders.

**Why:** notebooks cannot be unit-tested, reviewed, or versioned as a deployable
artifact. The wheel built from `src/` is the single deployment unit; everything else
orchestrates it.

## 2. Dependency and environment management

**Practice: one lockfile, exact pins for runtime, one source of truth for the version.**

- `pyproject.toml` is the single source of dependencies; a committed `uv.lock` (or
  equivalent lockfile) makes environments reproducible.
- Runtime dependencies are pinned exactly (`mlflow==3.11.1`); tooling in extras uses
  ranges (`pytest>=8.3, <9`). See [mlops/pyproject.toml](../mlops/pyproject.toml).
- Python itself is pinned to a minor version (`requires-python = ">=3.12, <3.13"`) and
  matches the Databricks runtime / serverless environment version used in the bundle.
- Optional dependency groups separate concerns: `dev` (local work: databricks-connect,
  ipykernel, pre-commit, pytest) and `ci` (the lean subset CI needs).
- The package version is single-sourced from `version.txt` via
  `dynamic = ["version"]`, so the wheel name is predictable wherever it is needed
  (for example when injecting the wheel into a model's `code_paths`).
- Never depend on standalone `pyspark`: it comes transitively via
  `databricks-connect` (dev/ci extra). Installing both breaks silently.

## 3. Configuration management

**Practice: typed, validated, environment-layered config; zero hardcoded names.**

- One `project_config.yml` with shared keys plus one block per environment
  (`dev` / `acc` / `prd`) holding what differs: catalog, schema, endpoint names, ids.
- A pydantic model loads it: `ProjectConfig.from_yaml(path, env)` validates that `env`
  is one of the known environments, merges the env block over the shared keys, and
  validates types and bounds (hyperparameters have `gt`/`le` constraints). See
  [mlops/src/hotel_booking/config.py](../mlops/src/hotel_booking/config.py) and
  [llmops/src/arxiv_curator/config.py](../llmops/src/arxiv_curator/config.py).
- No catalog, schema, workspace URL, endpoint name, or user name is hardcoded in
  `src/`, `scripts/`, or job YAML. Table names are always composed as
  `f"{cfg.catalog}.{cfg.schema}.<table>"`.
- The environment flows from the deployment, not from code: bundle target
  (`${bundle.target}`) → job parameter or widget `env` → `from_yaml(..., env)`.
- Secrets never appear in config files or code: Databricks secret scopes,
  `{{secrets/scope/key}}` references, or CI-injected environment variables only.

## 4. Code quality

**Practice: automated, pinned, enforced in CI, not left to discipline.**

- `.pre-commit-config.yaml` with pinned hook versions: basic hygiene hooks
  (large files, YAML/TOML/JSON validity, trailing whitespace) plus `ruff` (with
  `--fix --exit-non-zero-on-fix`) and `ruff-format`.
- Ruff configured in `pyproject.toml` with an explicit line length and a meaningful
  rule set: `F, E, W` plus `B` (bugbear), `I` (import sorting), `UP` (pyupgrade),
  `SIM` (simplification), `ERA` (dead code), `ANN` (type annotations required).
- Type annotations throughout `src/` using modern syntax (`str | None`, `list[str]`).
  Suppressions are local (file-level `# ruff: noqa: <codes>` with explicit codes) and
  justified, not blanket config-wide ignores.
- Structured logging (`loguru` or `logging`), never bare `print` in library code.
- Custom exception types for domain failures (for example
  `DeploymentNotApprovedError`) instead of generic `Exception`.

## 5. Testing

**Practice: a three-layer ladder; mock the boundary you don't own, never your own logic.**

The strategy is documented in the repo itself, see
[mlops/tests/README.md](../mlops/tests/README.md):

1. **Pure unit tests**: no mocks, no network. Config parsing, transformations,
   argument parsers.
2. **Unit tests with mocked boundaries**: Spark, `WorkspaceClient`, mlflow, vector
   search clients are mocked; the project's own logic is exercised for real. Key
   rules:
   - `autospec=True` on every patch of a real callable, so an SDK rename or signature
     change fails the test instead of passing silently.
   - Where autospec cannot work (dynamically-built clients like `WorkspaceClient`),
     build requests from real SDK dataclasses and enums so constructors validate
     field names at test time.
   - Both branches of idempotent code are tested (create-vs-update endpoint,
     create-vs-refresh monitor, exists-vs-not-exists index).
3. **Integration tests as Databricks jobs**, not pytest: a bundle-deployed job that
   chains the real production jobs end to end (see
   [llmops/resources/integration_test.yml](../llmops/resources/integration_test.yml)),
   run against a dev/acc workspace.

- `conftest.py` fixtures are Spark-free and network-free so the suite runs on a
  laptop and a bare CI runner.
- Fixtures include realistic edge cases (for example an invalid `2018-2-29` date row
  that the cleaning code must handle).
- Tests run through the locked environment: `uv run pytest`.

## 6. CI/CD

**Practice: test on PR, deploy on merge, no long-lived credentials.**

- **CI** (on pull request): install the `ci` extra from the lockfile, run
  `pre-commit run --all-files`, run `pytest`. Path filters scope each pipeline to its
  project folder in a monorepo (use `<folder>/**` patterns).
- **CD** (on merge to main): a matrix over environments (`acc`, `prd`) bound to
  GitHub Environments, so per-env approvals and variables apply.
- Authentication is short-lived: OIDC (`id-token: write`) to the cloud, secrets
  fetched at run time from a secret manager, service principal credentials for
  Databricks. No personal access tokens in repo secrets.
- Supply-chain hygiene: every third-party GitHub Action pinned to a commit SHA, CLI
  versions pinned.
- The deploy passes traceability into the bundle:
  `databricks bundle deploy --var="git_sha=${{ github.sha }}"`, so every job run and
  model version can be traced back to a commit.
- Deployment is only ever done by CI/CD for shared targets; humans deploy only to
  their own `dev` target.

## 7. Declarative Automation Bundles: everything as code

**Practice: one bundle per project; all jobs, alerts, warehouses, and dashboards are
declared, never hand-created in the UI.**

See [mlops/databricks.yml](../mlops/databricks.yml) and
[llmops/databricks.yml](../llmops/databricks.yml).

- Targets: `dev` (default, `mode: development`, per-user
  `root_path` under `/Workspace/Users/${workspace.current_user.userName}/...`),
  `acc`, and `prd` (shared root paths). Development artifacts never collide between
  users; shared environments never depend on a user's home folder.
- The wheel is the deploy unit: `artifacts: { type: whl, build: uv build }`, and every
  job depends on `../dist/*.whl` (serverless `environments` block or classic
  `libraries`). Jobs never `pip install` from git or copy source files around.
- Bundle variables carry deploy-time facts: `git_sha`, `branch`,
  `schedule_pause_status`, `usage_policy_id`. Every `${...}` reference is quoted.
- Schedules exist in every target but are `PAUSED` everywhere except `prd`
  (a `schedule_pause_status` variable overridden per target). Dev and acc deploys
  never fire production schedules.
- Jobs pass context into their tasks: `--env ${bundle.target}`,
  `--root_path ${workspace.root_path}`, `--git_sha ${var.git_sha}`, and job-runtime
  values like `{{job.run_id}}`, so runs are traceable and config resolves per env.
- Multi-step pipelines are a task DAG with explicit `depends_on`, and conditional
  steps use `condition_task` (for example: deploy only when training produced a
  better model). Cross-job orchestration uses `run_job_task`.

## 8. Cost attribution (FinOps)

**Practice: every piece of compute is attributable to the project in
`system.billing.usage`.**

- Serverless jobs and endpoints carry `budget_policy_id` (or `usage_policy_id` for
  online stores) referencing a bundle variable. A tags block does NOT work for
  serverless: an auto-assigned usage policy wins the key collision and the tag is
  silently dropped, so the policy must be set explicitly.
- Classic-compute jobs carry an explicit `tags: { project: <name> }`.
- SQL warehouses carry `custom_tags` with the project key.
- The policy id is defined once (bundle variable, mirrored in `project_config.yml`)
  rather than scattered through the code.
- Cost is actually monitored: a query or notebook against `system.billing.usage`
  filtered to the project's tags/policies and relevant SKUs. See
  [llmops/notebooks/chapter9_5.cost_monitoring.py](../llmops/notebooks/chapter9_5.cost_monitoring.py).
