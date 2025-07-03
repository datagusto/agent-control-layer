"""Tests for layer module."""

from unittest.mock import patch

import pytest

from agent_control_layer.layer import (
    SAFE_GLOBALS,
    _evaluate_contract,
    _is_rule_triggered,
    control_layer,
)


class TestIsRuleTriggered:
    """Test the _is_rule_triggered function."""

    def test_simple_expression_true(self):
        """Test simple expression that evaluates to True."""
        rule = {"trigger_condition": "len(tool_output) > 0"}
        tool_output = "test"
        assert _is_rule_triggered(rule, tool_output) is True

    def test_simple_expression_false(self):
        """Test simple expression that evaluates to False."""
        rule = {"trigger_condition": "len(tool_output) > 5"}
        tool_output = "test"
        assert _is_rule_triggered(rule, tool_output) is False

    def test_dict_tool_output(self):
        """Test with dictionary tool output."""
        rule = {
            "trigger_condition": "isinstance(tool_output, dict) and 'key' in tool_output"
        }
        tool_output = {"key": "value"}
        assert _is_rule_triggered(rule, tool_output) is True

    def test_list_tool_output(self):
        """Test with list tool output."""
        rule = {
            "trigger_condition": "isinstance(tool_output, list) and len(tool_output) == 2"
        }
        tool_output = ["item1", "item2"]
        assert _is_rule_triggered(rule, tool_output) is True

    def test_complex_expression(self):
        """Test complex expression with multiple conditions."""
        rule = {
            "trigger_condition": "isinstance(tool_output, dict) and tool_output.get('status') == 'error'"
        }
        tool_output = {"status": "error", "message": "failed"}
        assert _is_rule_triggered(rule, tool_output) is True

    def test_safe_globals_usage(self):
        """Test that safe globals are available in expression."""
        rule = {"trigger_condition": "max([1, 2, 3]) == 3"}
        tool_output = "test"
        assert _is_rule_triggered(rule, tool_output) is True

    def test_unsafe_expression_blocked(self):
        """Test that unsafe expressions are blocked."""
        rule = {"trigger_condition": "__import__('os').system('ls')"}
        tool_output = "test"

        with pytest.warns(UserWarning, match="Error evaluating expression"):
            result = _is_rule_triggered(rule, tool_output)
            assert result is False

    def test_invalid_expression_syntax(self):
        """Test handling of invalid expression syntax."""
        rule = {"trigger_condition": "invalid syntax here"}
        tool_output = "test"

        with pytest.warns(UserWarning, match="Error evaluating expression"):
            result = _is_rule_triggered(rule, tool_output)
            assert result is False

    def test_expression_with_undefined_variable(self):
        """Test handling of expression with undefined variable."""
        rule = {"trigger_condition": "undefined_var > 0"}
        tool_output = "test"

        with pytest.warns(UserWarning, match="Error evaluating expression"):
            result = _is_rule_triggered(rule, tool_output)
            assert result is False

    def test_none_trigger_condition(self):
        """Test handling when trigger_condition is None."""
        rule = {"trigger_condition": None}
        tool_output = "test"

        with pytest.warns(UserWarning, match="Error evaluating expression"):
            result = _is_rule_triggered(rule, tool_output)
            assert result is False

    def test_missing_trigger_condition(self):
        """Test handling when trigger_condition is missing."""
        rule = {}
        tool_output = "test"

        with pytest.warns(UserWarning, match="Error evaluating expression"):
            result = _is_rule_triggered(rule, tool_output)
            assert result is False


class TestEvaluateContract:
    """Test the _evaluate_contract function."""

    def test_no_rules(self):
        """Test contract with no rules."""
        contract = {"tool_name": "test_tool", "rules": []}
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        assert result == {"instruction": None, "rule": None}

    def test_missing_rules_key(self):
        """Test contract with missing rules key."""
        contract = {"tool_name": "test_tool"}
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        assert result == {"instruction": None, "rule": None}

    def test_first_rule_triggered(self):
        """Test when first rule is triggered."""
        contract = {
            "tool_name": "test_tool",
            "rules": [
                {
                    "trigger_condition": "len(tool_output) > 0",
                    "instruction": "first_instruction",
                },
                {
                    "trigger_condition": "len(tool_output) > 10",
                    "instruction": "second_instruction",
                },
            ],
        }
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        expected_rule = {
            "trigger_condition": "len(tool_output) > 0",
            "instruction": "first_instruction",
        }
        assert result == {"instruction": "first_instruction", "rule": expected_rule}

    def test_second_rule_triggered(self):
        """Test when second rule is triggered (first is not)."""
        contract = {
            "tool_name": "test_tool",
            "rules": [
                {
                    "trigger_condition": "len(tool_output) > 10",
                    "instruction": "first_instruction",
                },
                {
                    "trigger_condition": "len(tool_output) > 0",
                    "instruction": "second_instruction",
                },
            ],
        }
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        expected_rule = {
            "trigger_condition": "len(tool_output) > 0",
            "instruction": "second_instruction",
        }
        assert result == {"instruction": "second_instruction", "rule": expected_rule}

    def test_no_rule_triggered(self):
        """Test when no rule is triggered."""
        contract = {
            "tool_name": "test_tool",
            "rules": [
                {
                    "trigger_condition": "len(tool_output) > 10",
                    "instruction": "first_instruction",
                },
                {
                    "trigger_condition": "len(tool_output) > 20",
                    "instruction": "second_instruction",
                },
            ],
        }
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        assert result == {"instruction": None, "rule": None}

    def test_rule_priority_order(self):
        """Test that rules are evaluated in order (priority should be handled by config)."""
        contract = {
            "tool_name": "test_tool",
            "rules": [
                {
                    "trigger_condition": "len(tool_output) > 0",
                    "instruction": "high_priority",
                },
                {
                    "trigger_condition": "len(tool_output) > 0",
                    "instruction": "low_priority",
                },
            ],
        }
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        expected_rule = {
            "trigger_condition": "len(tool_output) > 0",
            "instruction": "high_priority",
        }
        assert result == {"instruction": "high_priority", "rule": expected_rule}

    def test_rule_without_instruction(self):
        """Test rule without instruction key."""
        contract = {
            "tool_name": "test_tool",
            "rules": [{"trigger_condition": "len(tool_output) > 0"}],
        }
        tool_output = "test"

        result = _evaluate_contract(contract, tool_output)
        expected_rule = {"trigger_condition": "len(tool_output) > 0"}
        assert result == {"instruction": None, "rule": expected_rule}


class TestControlLayer:
    """Test the control_layer function."""

    @patch("agent_control_layer.layer._config")
    def test_existing_tool_config(self, mock_config):
        """Test control layer with existing tool configuration."""
        mock_contract = {
            "tool_name": "test_tool",
            "rules": [
                {
                    "trigger_condition": "len(tool_output) > 0",
                    "instruction": "test_instruction",
                }
            ],
        }
        mock_config.get.return_value = mock_contract

        result = control_layer("test_tool", "test_output")

        mock_config.get.assert_called_once_with("test_tool")
        expected_rule = {
            "trigger_condition": "len(tool_output) > 0",
            "instruction": "test_instruction",
        }
        assert result == {"instruction": "test_instruction", "rule": expected_rule}

    @patch("agent_control_layer.layer._config")
    def test_non_existing_tool_config(self, mock_config):
        """Test control layer with non-existing tool configuration."""
        mock_config.get.return_value = None

        result = control_layer("non_existing_tool", "test_output")

        mock_config.get.assert_called_once_with("non_existing_tool")
        assert result == {"instruction": None, "rule": None}

    @patch("agent_control_layer.layer._config")
    def test_different_tool_output_types(self, mock_config):
        """Test control layer with different tool output types."""
        mock_contract = {
            "tool_name": "test_tool",
            "rules": [
                {
                    "trigger_condition": "isinstance(tool_output, dict)",
                    "instruction": "dict_instruction",
                },
                {
                    "trigger_condition": "isinstance(tool_output, list)",
                    "instruction": "list_instruction",
                },
                {
                    "trigger_condition": "isinstance(tool_output, str)",
                    "instruction": "str_instruction",
                },
            ],
        }
        mock_config.get.return_value = mock_contract

        # Test with dict
        result = control_layer("test_tool", {"key": "value"})
        expected_rule = {
            "trigger_condition": "isinstance(tool_output, dict)",
            "instruction": "dict_instruction",
        }
        assert result == {"instruction": "dict_instruction", "rule": expected_rule}

        # Test with list
        result = control_layer("test_tool", ["item1", "item2"])
        expected_rule = {
            "trigger_condition": "isinstance(tool_output, list)",
            "instruction": "list_instruction",
        }
        assert result == {"instruction": "list_instruction", "rule": expected_rule}

        # Test with string
        result = control_layer("test_tool", "test_string")
        expected_rule = {
            "trigger_condition": "isinstance(tool_output, str)",
            "instruction": "str_instruction",
        }
        assert result == {"instruction": "str_instruction", "rule": expected_rule}


class TestSafeGlobals:
    """Test the SAFE_GLOBALS dictionary."""

    def test_safe_globals_contains_expected_functions(self):
        """Test that SAFE_GLOBALS contains expected safe functions."""
        expected_functions = [
            "len",
            "all",
            "any",
            "isinstance",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "min",
            "max",
            "sum",
            "abs",
        ]

        for func_name in expected_functions:
            assert func_name in SAFE_GLOBALS
            assert callable(SAFE_GLOBALS[func_name])

    def test_safe_globals_does_not_contain_dangerous_functions(self):
        """Test that SAFE_GLOBALS does not contain dangerous functions."""
        dangerous_functions = ["__import__", "exec", "eval", "open", "input", "compile"]

        for func_name in dangerous_functions:
            assert func_name not in SAFE_GLOBALS
