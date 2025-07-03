import warnings
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ValidationError


class Rule(BaseModel):
    """Rule for a tool."""

    name: str
    description: str
    trigger_condition: str
    instruction: str
    priority: int


class Contract(BaseModel):
    """Contract for a tool."""

    tool_name: str
    description: str
    rules: list[Rule]


class Config:
    """Datagusto Agent Control Layer Config."""

    def __init__(self):
        """Initialize the config."""
        self.contracts: dict[str, dict[str, Any]] = {}
        self._load_config()

    def _load_config(self):
        """Load the config."""
        config_dir = Path.cwd() / ".dg_acl"

        if not config_dir.exists():
            warnings.warn(f"No config directory found at {config_dir}", stacklevel=2)
            return

        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        if not yaml_files:
            warnings.warn(f"No config files found in {config_dir}", stacklevel=2)
            return

        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    config_data = yaml.safe_load(f)
                    contract = Contract(**config_data)
                    tool_name = contract.tool_name

                    # sort rules by priority
                    sorted_rules = sorted(
                        contract.rules,
                        key=lambda x: x.priority,
                    )
                    contract.rules = sorted_rules

                    self.contracts[tool_name] = contract.model_dump()
            except ValidationError as e:
                warnings.warn(f"Invalid config file {yaml_file}: {e}", stacklevel=2)
                continue
            except Exception as e:
                warnings.warn(
                    f"Error loading config file {yaml_file}: {e}", stacklevel=2
                )
                continue

    def get(self, tool_name: str) -> Optional[dict[str, Any]]:
        """Get the config for a tool."""
        return self.contracts.get(tool_name)


_config = Config()
