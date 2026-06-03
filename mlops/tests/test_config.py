"""Unit tests for config parsing and validation (pure logic, no mocks)."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest
from pydantic import ValidationError

from hotel_booking.config import Parameters, ProjectConfig, Tags


def test_from_yaml_resolves_env_specific_catalog_and_schema(cfg: ProjectConfig) -> None:
    # dev block in project_config.yml -> mlops_dev / hotel_booking
    assert cfg.catalog == "mlops_dev"
    assert cfg.schema == "hotel_booking"


def test_from_yaml_pops_all_env_blocks(tmp_path, cfg: ProjectConfig) -> None:
    # The resolved config must not leak the raw env keys onto the model.
    dumped = cfg.model_dump()
    assert "dev" not in dumped
    assert "acc" not in dumped
    assert "prd" not in dumped


def test_from_yaml_rejects_unknown_env() -> None:
    with pytest.raises(ValueError, match="Invalid environment"):
        ProjectConfig.from_yaml(config_path="ignored.yml", env="staging")


def test_parameters_enforces_learning_rate_upper_bound() -> None:
    # learning_rate is bounded gt=0, le=0.2
    with pytest.raises(ValidationError):
        Parameters(learning_rate=0.5, n_estimators=200, max_depth=3)


def test_parameters_accepts_valid_values() -> None:
    p = Parameters(learning_rate=0.05, n_estimators=200, max_depth=3)
    assert p.n_estimators == 200


def test_tags_to_dict_omits_run_id_when_none() -> None:
    tags = Tags(git_sha="abc123", branch="main")
    assert tags.to_dict() == {"git_sha": "abc123", "branch": "main"}


def test_tags_to_dict_includes_run_id_when_set() -> None:
    tags = Tags(git_sha="abc123", branch="main", run_id="run-42")
    assert tags.to_dict()["run_id"] == "run-42"
