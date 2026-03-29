Deploy and run a notebook on Databricks using Databricks Asset Bundles.

The argument is the notebook path relative to the repo root (e.g. `llmops/notebooks/hello_world.py`).

This repo contains two independent Databricks projects — `llmops/` and `mlops/`. Each has its own `databricks.yml`, `resources/`, and wheel artifact. Determine which project the notebook belongs to from its path prefix (`llmops/` or `mlops/`). If unclear, ask the user. All bundle commands and relative paths below operate from within the project directory (i.e. after `cd llmops` or `cd mlops`).

## Steps

### 1. Determine the project and notebook relative path

From the notebook path, determine the project directory (`llmops` or `mlops`) and compute the notebook path relative to that project directory.

Example: `llmops/notebooks/hello_world.py` → project dir `llmops`, relative path `notebooks/hello_world.py`.

### 2. Find or derive the job resource key

From the project directory, scan all `resources/*.yml` files for any job whose `notebook_path` matches the target notebook. If a match is found, use that file's resource key and skip to step 6.

If no match is found, derive a resource key from the notebook filename: take the filename without extension, replace hyphens with underscores, append `_job`. Ignore leading numbers in the notebook name.

Example: `notebooks/hello_world.py` → resource key `hello_world_job`.

### 3. Validate `resources/` folder exists and is included in `databricks.yml`

Check that a `resources/` directory exists in the project directory. If not, create it.

Check that `databricks.yml` in the project directory includes the resources folder. It must contain an `include` block like:

```yaml
include:
  - resources/*.yml
```

If the `include` block is missing, add it to `databricks.yml` directly under the `bundle:` section.

### 4. Check for an existing job resource file

Look for `resources/<resource_key>.yml` in the project directory. If it already exists, skip creation and go to step 6.

### 5. Create the job resource file

First, read existing resource files in the project's `resources/` directory to determine the project's `tags.project_name` value:
- `llmops` uses `project_name: "arxiv-curator"`
- `mlops` uses `project_name: "hotel-booking"`

Create `resources/<resource_key>.yml` with this exact structure (substitute `<resource_key>`, `<job_display_name>`, `<project_name>`, and `<notebook_path>` with the actual values):

```yaml
resources:
  jobs:
    <resource_key>:
      name: <job_display_name>
      tags:
        project_name: "<project_name>"

      environments:
        - environment_key: default
          spec:
            environment_version: "4"
            dependencies:
              - ../dist/*.whl

      tasks:
        - task_key: run_notebook
          environment_key: default
          notebook_task:
            notebook_path: <notebook_path>
            base_parameters:
              env: ${bundle.target}
              git_sha: "${var.git_sha}"
              run_id: "{{job.run_id}}"
```

Where:
- `<resource_key>` = the derived key (e.g. `hello_world_job`)
- `<job_display_name>` = kebab-case version of the notebook name (e.g. `hello-world`)
- `<project_name>` = the tag from existing resources (`arxiv-curator` or `hotel-booking`)
- `<notebook_path>` = notebook path relative to the project directory (e.g. `notebooks/hello_world.py`)

Also add the `git_sha` variable definition to the project's `databricks.yml` if it is not already present:

```yaml
variables:
  git_sha:
    description: "Git SHA of the deployed commit"
    default: "local"
```

### 6. Deploy the bundle

`cd` into the project directory and run:
```bash
cd <project_dir>
databricks bundle deploy
```

### 7. Run the job

From the same project directory, run:
```bash
databricks bundle run <resource_key>
```

Where `<resource_key>` is the derived key from step 2.

Report the output to the user, including any run URL printed by the CLI.

## Example

```bash
/run-notebook llmops/notebooks/hello_world.py
```

This detects the `llmops` project, derives resource key `hello_world_job`, creates `llmops/resources/hello_world_job.yml` with `project_name: "arxiv-curator"`, then runs `cd llmops && databricks bundle deploy && databricks bundle run hello_world_job`.
