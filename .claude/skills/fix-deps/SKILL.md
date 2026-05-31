---
name: fix-deps
description: Update every dependency in pyproject.toml to its latest stable version from PyPI and resync the environment with uv. Trigger whenever the user asks to fix/update/bump/refresh dependencies, sync deps to latest, upgrade pyproject.toml versions, or pin packages to the newest stable releases.
---

# fix-deps

Look up the latest stable version of every dependency in `pyproject.toml` using the PyPI JSON API (`https://pypi.org/pypi/<package>/json`). Strip any extras from the package name before querying (e.g. `databricks-sdk[notebook]` → query `databricks-sdk`, keep the `[notebook]` in the final specifier).

Apply these pinning rules when updating `pyproject.toml`:

**Regular dependencies** (`[project] dependencies`):
- Pin to exact version: `==X.Y.Z`

**Optional / dev dependencies** (`[project.optional-dependencies]` and `[dependency-groups]`):
- Use `>=X.Y.Z,<NEXT_MAJOR` where NEXT_MAJOR is the next integer major version

**Packages that must always stay optional** (never in regular `dependencies`):
- `databricks-connect` — group: `notebook`
- `ipykernel` — group: `notebook`
- `pytest` — group: `dev`
- `pytest-cov` — group: `dev`
- `pre-commit` — group: `dev`
- `ruff` — group: `dev`
- `ty` — group: `dev`

After determining all versions, update `pyproject.toml` in place. Keep `[project.optional-dependencies]` and `[dependency-groups]` in sync with the same packages and constraints.

Finally, validate the environment resolves correctly by running:
```bash
uv sync --extra dev
```
