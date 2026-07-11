# LLMOps track: GenAI / agent lifecycle practices

Reference implementation: [llmops/](../llmops/) (arxiv paper curator agent, MLflow
ResponsesAgent + managed MCP + vector search + agent framework deployment).
Foundations ([foundations.md](foundations.md)) apply on top of everything here.

## 1. Agent architecture

**Practice: the agent is a versioned, packaged, configurable artifact, not a notebook
cell.**

- The agent class lives in the package
  ([llmops/src/arxiv_curator/agent.py](../llmops/src/arxiv_curator/agent.py)) and
  implements a standard serving contract (`mlflow.pyfunc.ResponsesAgent` with
  `predict` and `predict_stream`).
- Logging uses **models-from-code**: a small root-level entry file
  ([llmops/arxiv_agent.py](../llmops/arxiv_agent.py)) that builds the agent from
  `mlflow.models.ModelConfig` and calls `mlflow.models.set_model(agent)`. The real
  logic stays in the wheel; the entry file is the artifact MLflow serializes.
- Configuration (LLM endpoint, catalog/schema, Genie space, system prompt) is
  injected via `model_config` at log time and stored with the model version. The
  system prompt lives in `project_config.yml`, versioned with the code, never
  hardcoded in the class.

## 2. Tools and robustness

**Practice: assume the LLM and its tools will misbehave; bound every loop and guard
every boundary.**

- Tools are discovered and wrapped uniformly (managed MCP servers for vector search
  and Genie; each tool carries its spec plus an `exec_fn`). See
  [llmops/src/arxiv_curator/mcp.py](../llmops/src/arxiv_curator/mcp.py).
- The tool loop has a hard `max_iter` cap and emits an explicit "max iterations
  reached" message instead of looping forever.
- Unknown tool names return an error string to the model rather than raising.
- LLM calls are wrapped with exponential backoff on rate-limit errors.
- Stateful extras degrade gracefully: memory (Lakebase/Postgres) failures log a
  warning and return empty history; they never crash a prediction. Connection
  pooling resets on stale short-lived credentials. See
  [llmops/src/arxiv_curator/memory.py](../llmops/src/arxiv_curator/memory.py).

## 3. Tracing and observability

**Practice: every production request produces a trace that can be tied to a commit, a
model version, and a session.**

- Spans are typed: `@mlflow.trace` with `SpanType.AGENT` / `CHAIN` / `TOOL` /
  `RETRIEVER`, plus a manual `LLM` span that records token usage and the provider
  request id.
- Each trace is stamped with `git_sha`, `model_version`, and the serving endpoint
  name, read from environment variables injected at deploy time. Lineage runs from a
  billing row or a trace back to the exact commit.
- Session and request correlation: callers pass `session_id` / `client_request_id`
  in `custom_inputs`, and the agent writes them into trace metadata, so
  multi-turn sessions can be reconstructed from the trace table.

## 4. Evaluation, before and after deployment

**Practice: evaluation gates registration; the same scorers keep running in
production.**

- An evaluation dataset lives in the repo
  ([llmops/eval_inputs.txt](../llmops/eval_inputs.txt)) and evolves with the agent.
- Scorers mix cheap deterministic checks (word count) with LLM judges
  (`Guidelines` scorers for tone and structure) via `mlflow.genai.evaluate`. See
  [llmops/src/arxiv_curator/evaluation.py](../llmops/src/arxiv_curator/evaluation.py).
- Evaluation runs **before** log/register, and the resulting metrics are logged onto
  the model's run, so every registered version carries its eval scores.
- In production, a scheduled job scores **only new traces** (those without
  assessments), runs cheap scorers on everything and expensive LLM judges on a
  sample, and writes results back with `mlflow.log_feedback`. See
  [llmops/resources/deployment_notebooks/production_monitoring.py](../llmops/resources/deployment_notebooks/production_monitoring.py).

## 5. Registration and deployment

**Practice: declare dependencies at log time, promote by alias, and never mix the two
auth models.**

- Every Databricks resource the agent touches is declared in `resources=[...]` at log
  time (`DatabricksServingEndpoint`, `DatabricksVectorSearchIndex`,
  `DatabricksGenieSpace`, `DatabricksTable`, `DatabricksSQLWarehouse`), so the
  deployment provisions auth to exactly those resources.
- Registration goes to Unity Catalog with provenance tags (`git_sha`, `run_id`) and
  promotion uses the `latest-model` alias; deployment resolves the version by alias.
- Deployment (`agents.deploy`) sets `scale_to_zero`, a deliberate workload size, and
  `budget_policy_id` for cost attribution.
- `environment_vars` carry only app-level values (`GIT_SHA`, `MODEL_VERSION`,
  `MODEL_SERVING_ENDPOINT_NAME`, `MLFLOW_EXPERIMENT_ID`) and secret **references**
  (`{{secrets/scope/key}}`), never plaintext secrets.
- Auth is one of two modes, never mixed: either the agent framework owns auth (no
  `DATABRICKS_CLIENT_ID`/`SECRET`/`HOST` in env vars) or an explicit SPN owns it
  everywhere. Setting the SPN variables on top of framework auth silently breaks
  access to the declared resources.
- Dedicated service principals are created as code, their secrets stored in a secret
  scope, and least-privilege grants applied (for example specific Postgres roles and
  table privileges for memory). Rotation is deploy-aware: secrets resolve at deploy
  time, so rotate, redeploy, then invalidate.

## 6. Vector search and data pipeline

**Practice: retrieval infrastructure is idempotent and readiness-aware.**

- Endpoint and index creation is guarded by existence checks; creation waits for the
  resource to be ready before syncing or querying (async provisioning otherwise
  bites). See
  [llmops/src/arxiv_curator/vector_search.py](../llmops/src/arxiv_curator/vector_search.py).
- The index is a Delta Sync index over a source table with Change Data Feed enabled;
  syncs are triggered incrementally by the scheduled data pipeline.
- Document processing (download, parse, chunk, clean) is package code with its own
  job, not notebook-only logic.

## 7. Production monitoring, dashboard, cost

**Practice: the trace table is a data asset; aggregate it, visualize it, alert on it,
and watch the bill.**

- A scheduled job maintains an aggregated view over the trace table: latency, LLM and
  tool-call counts per trace, total tokens, and quality scores.
- A Lakeview dashboard over that view, plus its dedicated small auto-stopping
  warehouse, are declared as bundle resources.
- Cost monitoring queries `system.billing.usage` for the GenAI SKUs (model serving,
  vector search, foundation model APIs, AI gateway, Lakebase), filtered by the
  project's usage policy tags.

## 8. Integration testing as a bundle job

**Practice: test the wiring by running the real jobs, chained.**

[llmops/resources/integration_test.yml](../llmops/resources/integration_test.yml)
chains the actual production jobs (process data → register and deploy → monitoring)
with `run_job_task` and `depends_on`, parameterized with an `integration_testing`
flag. It reuses the production definitions instead of a parallel test-only script, so
the test exercises exactly what will run in prd.
