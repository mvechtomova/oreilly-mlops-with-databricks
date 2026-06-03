"""Shared fixtures for the arxiv-curator test suite.

These fixtures avoid any Spark, Databricks, or network dependency so the unit
layer runs on a plain GitHub runner. Anything that needs a real runtime belongs
in tests/integration and is marked with @pytest.mark.integration.
"""

import pytest

from arxiv_curator.config import ProjectConfig


@pytest.fixture
def cfg() -> ProjectConfig:
    """The real project config, resolved for the dev environment.

    pytest's rootdir is the llmops/ project (set via pyproject.toml), so the
    config path is relative to it.
    """
    return ProjectConfig.from_yaml(config_path="project_config.yml", env="dev")
