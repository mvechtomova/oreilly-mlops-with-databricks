"""Unit tests for ProjectConfig.from_yaml."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from arxiv_curator.config import ProjectConfig


def test_from_yaml_loads_dev_env(cfg) -> None:
    assert cfg.catalog == "mlops_dev"
    assert cfg.schema == "arxiv"
    # shared (non-env-scoped) keys are injected onto the env config
    assert cfg.system_prompt
    assert cfg.experiment_path == "/Shared/genai-arxiv-agent"


def test_from_yaml_rejects_unknown_env() -> None:
    with pytest.raises(ValueError, match="Invalid environment"):
        ProjectConfig.from_yaml(config_path="project_config.yml", env="staging")


def test_optional_keys_default_to_none(tmp_path) -> None:
    cfg_file = tmp_path / "project_config.yml"
    cfg_file.write_text(
        "system_prompt: hi\n"
        "experiment_path: /Shared/x\n"
        "dev:\n"
        "  catalog: c\n"
        "  schema: s\n"
        "  volume: v\n"
        "  genie_space_id: g\n"
        "  llm_endpoint: e\n"
        "  warehouse_id: w\n"
    )
    cfg = ProjectConfig.from_yaml(config_path=str(cfg_file), env="dev")
    assert cfg.usage_policy_id is None
    assert cfg.lakebase_project_id is None
