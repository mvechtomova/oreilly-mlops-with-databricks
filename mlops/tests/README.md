# Testing strategy

Three layers, each running as far "down" the cost ladder as it can. Push every
assertion to the cheapest layer that can hold it.

## Layer 1 - unit, no mocks

Pure transforms with no external dependency. Fast, deterministic, run anywhere.

- `test_config.py` - env resolution, pydantic validation bounds, `Tags.to_dict`.
- `test_data_processor.py` - column rename, the leap-year date fix, arrival-date math.
- `test_lightgbm_model.py` - pipeline trains and predicts; unknown categories -> -1.

## Layer 2 - unit with `pytest-mock`

Our logic wrapped around a client we don't own (Spark, `WorkspaceClient`,
`mlflow`, `DeltaTable`). Mock the boundary, assert our code drives it correctly.
Still no cluster required.

- `test_data_loader.py` - SQL window/version logic with a mocked Spark session.
- `test_serving.py` - the create-vs-update endpoint branch with a mocked WorkspaceClient.
- `test_common.py` - delta version read, MLflow URI env-branching, `adjust_price`.

Rule of thumb: mock at the boundary you don't own, never your own logic. A test
that mocks `DataProcessor` tests nothing.

## Layer 3 - integration (separate CI step, not pytest)

The real wiring against a live workspace. Catches what mocks can't: schema drift,
serving-contract mismatches, UC permissions. Slow and bills compute, so it runs
as its own CI pipeline step (a Databricks job run), not under pytest.

## Running

```bash
uv run pytest          # the unit + mocked layers; no cluster needed
```
