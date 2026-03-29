import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Configuration for the arxiv curator project."""
    catalog: str
    schema: str
    volume: str
    genie_space_id: str
    llm_endpoint: str
    system_prompt: str
    warehouse_id: str
    experiment_path: str
    usage_policy_id: str | None = None
    # Lakebase connection settings
    lakebase_project_id: str | None = None

    @classmethod
    def from_yaml(cls: "ProjectConfig",
                  config_path: str,
                  env: str = "dev") -> "ProjectConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the project_config.yml file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(
                f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'"
            )

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        env_config = config_dict[env]
        env_config["system_prompt"] = config_dict["system_prompt"]
        env_config["experiment_path"] = config_dict["experiment_path"]
        env_config["usage_policy_id"] = config_dict.get("usage_policy_id")
        env_config["lakebase_project_id"] = config_dict.get(
            "lakebase_project_id"
        )
        return cls(**env_config)
