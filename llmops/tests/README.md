# Testing strategy

Three layers, each running as far "down" the cost ladder as it can. Push every
assertion to the cheapest layer that can hold it.

## Layer 1 - unit, no mocks

Pure logic with no external dependency. Fast, deterministic, run anywhere.

- `test_config.py` - env resolution, shared-key injection, pydantic optionals.
- `test_exceptions.py` - the custom deployment exceptions are raisable.

## Layer 2 - unit with `pytest-mock`

Our logic wrapped around a client we don't own (`DatabricksMCPClient`,
`VectorSearchClient`, `WorkspaceClient`). Mock the boundary, assert our code
drives it correctly. No workspace required.

- `test_mcp.py` - the managed exec_fn joins tool output; `create_mcp_tools`
  builds the OpenAI tool spec from the MCP client.
- `test_vector_search.py` - the create-vs-get branch for the endpoint and index
  with a mocked `VectorSearchClient`.

Rule of thumb: mock at the boundary you don't own, never your own logic.

## Layer 3 - integration (separate CI step, not pytest)

The real wiring against a live workspace and a live LLM endpoint. Catches what
mocks can't: serving-contract drift, MCP/Genie permissions, vector-index sync.
Slow and bills compute, so it runs as its own CI pipeline step (a Databricks job
run), not under pytest.

## Running

```bash
uv run --extra dev pytest    # the unit + mocked layers; no cluster needed
```
