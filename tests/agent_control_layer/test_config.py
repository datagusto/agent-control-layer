import os
import shutil
import unittest
from pathlib import Path

import yaml

from agent_control_layer.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path("test_config_temp_dir")
        self.test_dir.mkdir(exist_ok=True)

        # Store the original working directory
        self.original_cwd = Path.cwd()
        os.chdir(self.test_dir)

        # Create the config directory
        self.config_dir = Path(".dg_acl")
        self.config_dir.mkdir()

    def tearDown(self):
        # Return to the original working directory
        os.chdir(self.original_cwd)
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def _write_config(self, filename: str, data: dict):
        """Helper function to write a config file."""
        with open(self.config_dir / filename, "w") as f:
            yaml.dump(data, f)

    def test_load_config_with_valid_yaml_and_rule_sorting(self):
        """Tests loading a valid config and ensures rules are sorted by priority."""
        config_data = {
            "tool_name": "search",
            "description": "Rules for the search tool",
            "rules": [
                {
                    "name": "rule_priority_2",
                    "description": "Rule with priority 2",
                    "trigger_condition": "condition2",
                    "instruction": "instruction2",
                    "priority": 2,
                },
                {
                    "name": "rule_priority_1",
                    "description": "Rule with priority 1",
                    "trigger_condition": "condition1",
                    "instruction": "instruction1",
                    "priority": 1,
                },
            ],
        }
        self._write_config("config.yaml", config_data)

        config = Config()
        self.assertIn("search", config.contracts)
        search_contract = config.get("search")
        self.assertIsNotNone(search_contract)

        rules = search_contract.get("rules", [])
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["priority"], 1)
        self.assertEqual(rules[0]["name"], "rule_priority_1")
        self.assertEqual(rules[1]["priority"], 2)
        self.assertEqual(rules[1]["name"], "rule_priority_2")

    def test_load_config_with_validation_error(self):
        """Tests that a file with a validation error (e.g., missing field) raises a warning."""
        # Config data missing 'priority' in a rule
        invalid_config_data = {
            "tool_name": "search",
            "description": "Rules for the search tool",
            "rules": [
                {
                    "name": "invalid_rule",
                    "description": "This rule is missing priority.",
                    "trigger_condition": "condition",
                    "instruction": "instruction",
                }
            ],
        }
        self._write_config("invalid_config.yaml", invalid_config_data)

        with self.assertWarnsRegex(UserWarning, "Invalid config file"):
            config = Config()

        # The invalid config should not be loaded
        self.assertEqual(len(config.contracts), 0)

    def test_load_config_with_no_config_dir(self):
        """Tests that a warning is raised when the .dg_acl directory does not exist."""
        shutil.rmtree(self.config_dir)
        with self.assertWarnsRegex(UserWarning, "No config directory found"):
            Config()

    def test_load_config_with_no_yaml_files(self):
        """Tests that a warning is raised when the config directory is empty."""
        with self.assertWarnsRegex(UserWarning, "No config files found"):
            Config()

    def test_get_config(self):
        """Tests the get() method for retrieving tool configurations."""
        config_data = {
            "tool_name": "search",
            "description": "Rules for the search tool",
            "rules": [
                {
                    "name": "r",
                    "description": "d",
                    "trigger_condition": "c",
                    "instruction": "i",
                    "priority": 1,
                }
            ],
        }
        self._write_config("config.yaml", config_data)

        config = Config()
        search_config = config.get("search")
        self.assertIsNotNone(search_config)
        self.assertEqual(search_config["tool_name"], "search")

        none_config = config.get("non_existent_tool")
        self.assertIsNone(none_config)

    def test_load_multiple_and_mixed_extension_files(self):
        """Tests loading multiple config files with .yaml and .yml extensions."""
        config_data_1 = {
            "tool_name": "tool1",
            "description": "Rules for tool1",
            "rules": [
                {
                    "name": "r1",
                    "description": "d",
                    "trigger_condition": "c",
                    "instruction": "i",
                    "priority": 1,
                }
            ],
        }
        config_data_2 = {
            "tool_name": "tool2",
            "description": "Rules for tool2",
            "rules": [
                {
                    "name": "r2",
                    "description": "d",
                    "trigger_condition": "c",
                    "instruction": "i",
                    "priority": 1,
                }
            ],
        }
        self._write_config("tool1.yaml", config_data_1)
        self._write_config("tool2.yml", config_data_2)

        config = Config()
        self.assertEqual(len(config.contracts), 2)
        self.assertIn("tool1", config.contracts)
        self.assertIn("tool2", config.contracts)

    def test_load_with_one_invalid_file_among_valid_ones(self):
        """Tests that valid configs are loaded even if one file is invalid."""
        valid_config = {
            "tool_name": "valid_tool",
            "description": "A valid tool config",
            "rules": [
                {
                    "name": "r",
                    "description": "d",
                    "trigger_condition": "c",
                    "instruction": "i",
                    "priority": 1,
                }
            ],
        }
        # Invalid because 'rules' is missing
        invalid_config = {"tool_name": "invalid_tool", "description": "d"}

        self._write_config("valid.yaml", valid_config)
        self._write_config("invalid.yml", invalid_config)

        with self.assertWarns(UserWarning):
            config = Config()

        self.assertEqual(len(config.contracts), 1)
        self.assertIn("valid_tool", config.contracts)
        self.assertNotIn("invalid_tool", config.contracts)

    def test_load_config_with_malformed_yaml_syntax(self):
        """Tests that a file with invalid YAML syntax raises a warning."""
        malformed_yaml = "tool_name: search\n- description: invalid syntax"
        with open(self.config_dir / "malformed.yaml", "w") as f:
            f.write(malformed_yaml)

        with self.assertWarnsRegex(UserWarning, "Error loading config file"):
            config = Config()

        self.assertEqual(len(config.contracts), 0)


if __name__ == "__main__":
    unittest.main()
