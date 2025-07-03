"""Tests for langgraph tools module."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_control_layer.langgraph.tools import (
    _find_latest_tool_message,
    build_control_layer_tools,
)


class TestFindLatestToolMessage:
    """Test the _find_latest_tool_message function."""

    def test_find_latest_tool_message_success(self):
        """Test finding the latest ToolMessage successfully."""
        messages = [
            HumanMessage(content="hi", id="1"),
            AIMessage(content="hello", id="2"),
            ToolMessage(content="first", name="tool1", tool_call_id="t1"),
            HumanMessage(content="another", id="3"),
            ToolMessage(content="second", name="tool2", tool_call_id="t2"),
        ]

        result = _find_latest_tool_message(messages)

        assert isinstance(result, ToolMessage)
        assert result.content == "second"
        assert result.name == "tool2"

    def test_find_latest_tool_message_single(self):
        """Test finding ToolMessage when only one exists."""
        messages = [
            HumanMessage(content="hi", id="1"),
            ToolMessage(content="only", name="tool1", tool_call_id="t1"),
            AIMessage(content="hello", id="2"),
        ]

        result = _find_latest_tool_message(messages)

        assert isinstance(result, ToolMessage)
        assert result.content == "only"
        assert result.name == "tool1"

    def test_find_latest_tool_message_not_found(self):
        """Test when no ToolMessage is found."""
        messages = [
            HumanMessage(content="hi", id="1"),
            AIMessage(content="hello", id="2"),
        ]

        result = _find_latest_tool_message(messages)

        assert result is None

    def test_find_latest_tool_message_empty_list(self):
        """Test with empty messages list."""
        messages = []

        result = _find_latest_tool_message(messages)

        assert result is None

    def test_find_latest_tool_message_only_tool_messages(self):
        """Test with list containing only ToolMessages."""
        messages = [
            ToolMessage(content="first", name="tool1", tool_call_id="t1"),
            ToolMessage(content="second", name="tool2", tool_call_id="t2"),
            ToolMessage(content="third", name="tool3", tool_call_id="t3"),
        ]

        result = _find_latest_tool_message(messages)

        assert isinstance(result, ToolMessage)
        assert result.content == "third"
        assert result.name == "tool3"


class TestBuildControlLayerTools:
    """Test the build_control_layer_tools function."""

    def test_build_control_layer_tools_returns_functions(self):
        """Test that build_control_layer_tools returns a list of functions."""
        MockStateClass = MagicMock()

        tools = build_control_layer_tools(MagicMock())

        assert isinstance(tools, list)
        assert len(tools) == 2
        assert callable(tools[0])
        assert callable(tools[1])

    @patch("agent_control_layer.langgraph.tools._config")
    def test_control_layer_init_function(self, mock_config):
        """Test the control_layer_init function."""
        mock_config.contracts.keys.return_value = ["tool1", "tool2"]
        mock_state = MagicMock()

        tools = build_control_layer_tools(MagicMock())
        control_layer_init = tools[0]

        result = control_layer_init(mock_state)

        expected_result = {
            "additional_instruction": "control_layer_post_hook must be invoked after every tool execution to ensure data governance compliance.",
            "target_tool_list": ["tool1", "tool2"],
        }
        assert result == expected_result

    def test_control_layer_post_hook_no_tool_message(self):
        """Test control_layer_post_hook when no ToolMessage is found."""
        mock_state = MagicMock()
        mock_state.messages = [HumanMessage(content="hi", id="1")]

        tools = build_control_layer_tools(MagicMock())
        control_layer_post_hook = tools[1]

        result = control_layer_post_hook(mock_state)

        assert result == {
            "additional_instruction": "No further instruction.",
            "triggered_rule": None,
        }

    @patch("agent_control_layer.langgraph.tools.control_layer")
    def test_control_layer_post_hook_with_tool_message(self, mock_control_layer):
        """Test control_layer_post_hook with ToolMessage."""
        mock_rule = {"description": "test rule description"}
        mock_control_layer.return_value = {
            "instruction": "test_instruction",
            "rule": mock_rule,
        }

        mock_state = MagicMock()
        tool_message = ToolMessage(
            content="output", name="test_tool", tool_call_id="t1"
        )
        mock_state.messages = [tool_message]

        tools = build_control_layer_tools(MagicMock())
        control_layer_post_hook = tools[1]

        result = control_layer_post_hook(mock_state)

        mock_control_layer.assert_called_once_with("test_tool", "output")

        expected_message = "[DATAGUSTO CONTROL LAYER DIRECTIVE] The tool test_tool's output does not satisfy the following policy: test rule description. Please follow the instruction: test_instruction"
        assert result == {
            "additional_instruction": expected_message,
            "triggered_rule": mock_rule,
        }

    @patch("agent_control_layer.langgraph.tools.control_layer")
    def test_control_layer_post_hook_with_json_content(self, mock_control_layer):
        """Test control_layer_post_hook with JSON content."""
        mock_rule = {"description": "json rule description"}
        mock_control_layer.return_value = {
            "instruction": "json_instruction",
            "rule": mock_rule,
        }
        json_output = json.dumps({"key": "value"})

        mock_state = MagicMock()
        tool_message = ToolMessage(
            content=json_output, name="json_tool", tool_call_id="t1"
        )
        mock_state.messages = [tool_message]

        tools = build_control_layer_tools(MagicMock())
        control_layer_post_hook = tools[1]

        result = control_layer_post_hook(mock_state)

        mock_control_layer.assert_called_once_with("json_tool", {"key": "value"})

        expected_message = "[DATAGUSTO CONTROL LAYER DIRECTIVE] The tool json_tool's output does not satisfy the following policy: json rule description. Please follow the instruction: json_instruction"
        assert result == {
            "additional_instruction": expected_message,
            "triggered_rule": mock_rule,
        }

    @patch("agent_control_layer.langgraph.tools.control_layer")
    def test_control_layer_post_hook_with_invalid_json(self, mock_control_layer):
        """Test control_layer_post_hook with invalid JSON content."""
        mock_rule = {"description": "invalid json rule description"}
        mock_control_layer.return_value = {
            "instruction": "invalid_json_instruction",
            "rule": mock_rule,
        }
        invalid_json_output = "{key: 'value'}"  # Invalid JSON

        mock_state = MagicMock()
        tool_message = ToolMessage(
            content=invalid_json_output, name="invalid_json_tool", tool_call_id="t1"
        )
        mock_state.messages = [tool_message]

        tools = build_control_layer_tools(MagicMock())
        control_layer_post_hook = tools[1]

        result = control_layer_post_hook(mock_state)

        mock_control_layer.assert_called_once_with(
            "invalid_json_tool", invalid_json_output
        )

        expected_message = "[DATAGUSTO CONTROL LAYER DIRECTIVE] The tool invalid_json_tool's output does not satisfy the following policy: invalid json rule description. Please follow the instruction: invalid_json_instruction"
        assert result == {
            "additional_instruction": expected_message,
            "triggered_rule": mock_rule,
        }

    @patch("agent_control_layer.langgraph.tools.control_layer")
    def test_control_layer_post_hook_with_mixed_messages(self, mock_control_layer):
        """Test control_layer_post_hook with mixed message types."""
        mock_rule = {"description": "mixed rule description"}
        mock_control_layer.return_value = {
            "instruction": "mixed_instruction",
            "rule": mock_rule,
        }
        mock_state = MagicMock()

        # Create mixed messages with ToolMessage being the latest
        messages = [
            HumanMessage(content="hi", id="1"),
            ToolMessage(content="first_tool", name="tool1", tool_call_id="t1"),
            AIMessage(content="AI says hi", id="2"),
            ToolMessage(content="second_tool", name="tool2", tool_call_id="t2"),
            HumanMessage(content="another", id="3"),
        ]
        mock_state.messages = messages

        tools = build_control_layer_tools(MagicMock())
        control_layer_post_hook = tools[1]

        result = control_layer_post_hook(mock_state)

        mock_control_layer.assert_called_once_with("tool2", "second_tool")
        assert "additional_instruction" in result
        assert "mixed_instruction" in result["additional_instruction"]
        assert "triggered_rule" in result
        assert result["triggered_rule"] == mock_rule

    def test_function_docstrings_exist(self):
        """Test that functions have proper docstrings."""
        MockStateClass = MagicMock()
        tools = build_control_layer_tools(MagicMock())

        control_layer_init = tools[0]
        control_layer_post_hook = tools[1]

        assert control_layer_init.__doc__ is not None
        assert "MANDATORY" in control_layer_init.__doc__
        assert "control_layer_init" in control_layer_init.__doc__

        assert control_layer_post_hook.__doc__ is not None
        assert "MANDATORY" in control_layer_post_hook.__doc__
        assert "control_layer_post_hook" in control_layer_post_hook.__doc__
