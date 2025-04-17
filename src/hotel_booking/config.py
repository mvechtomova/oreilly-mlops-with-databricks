"""Configuration file for the project."""

from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Project configuration parameters loaded from project_config.yml."""

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    parameters_a: dict[str, Any] = None
    parameters_b: dict[str, Any] = None
    experiment_name_basic: str = None
    experiment_name_custom: str = None
    experiment_name_fe: str = None
    pipeline_id: str = None

    @classmethod
    def from_yaml(cls: "ProjectConfig", config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config_dict["catalog_name"] = config_dict[env]["catalog_name"]
        config_dict["schema_name"] = config_dict[env]["schema_name"]
        if "pipeline_id" in config_dict[env]:
            config_dict["pipeline_id"] = config_dict[env]["pipeline_id"]
        for k in ["dev", "acc", "prd"]:
            config_dict.pop(k)
        return cls(**config_dict)
