import contextlib
import json
from typing import Annotated, Any, Optional

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState

from agent_control_layer.config import _config
from agent_control_layer.layer import control_layer


def _find_latest_tool_message(messages: list) -> Optional[ToolMessage]:
    """Find the latest tool message in the messages list."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message
    return None


def build_control_layer_tools(state_class):
    """Build the tools for the control layer."""

    def control_layer_init(
        state: Annotated[Any, InjectedState],
    ) -> dict[str, Any]:
        """MANDATORY: control_layer_init function that MUST be called exactly ONCE at
        the beginning of each new conversation (trace).

        ⚠️  CRITICAL REQUIREMENT: This function MUST be invoked as the very first Tool
        call when a new conversation or trace begins. Call this function ONCE and ONLY
        ONCE per conversation session. DO NOT call this function multiple times within
        the same conversation or trace.

        This function initializes the Dynamic Agent Controller that manages and controls
        AI agent behavior based on data quality from external data exchanges via Tools.
        The Dynamic Agent Controller monitors data governance policies and ensures
        proper handling of external data interactions throughout the conversation
        session.

        Returns:
            dict[str, Any]: A dictionary containing the names of all Tools that are
                subject to agent controller monitoring.
                The returned dictionary contains the names of all Tools that require
                data governance oversight during the conversation session.

                ⚠️  IMPORTANT: When ANY of the Tools functions listed in the return value
                are executed by the AI agent, the "control_layer_post_hook" tool MUST
                be called immediately after their execution. This ensures proper data
                governance and quality control for all monitored external data
                interactions.
        """
        return {
            "additional_instruction": (
                "control_layer_post_hook must be invoked after every tool execution "
                "to ensure data governance compliance."
            ),
            "target_tool_list": list(_config.contracts.keys()),
        }

    def control_layer_post_hook(
        state: Annotated[Any, InjectedState],
    ) -> dict[str, Any]:
        """MANDATORY: Control Layer Post-hook function that MUST be called after EVERY
        single Tool execution.

        ⚠️  CRITICAL REQUIREMENT: This function MUST be invoked immediately after EACH
        and EVERY Tool execution.
        DO NOT skip this call or batch multiple Tool/MCP executions before calling this
        function.
        Each Tool → control_layer_post_hook pattern must be strictly followed:
        (1) Execute any Tool, (2) IMMEDIATELY call control_layer_post_hook,
        (3) Follow the instructions from the control_layer_post_hook, (4) Repeat for
        each Tool execution.

        This function serves as the Control Layer that manages and controls AI agent
        behavior based on data quality from external data exchanges via Tools.
        It enforces data governance policies and ensures proper handling of external
        data interactions. Failure to call this function after every Tool execution
        will result in data governance violations and may cause system errors.

        Returns:
            dict[str, Any]: An additional instruction from the Control Layer to the AI
                agent. The instruction is generated based on the data quality assessment
                and governance policies applied to the Tool output.

                ⚠️  MANDATORY COMPLIANCE: The AI agent MUST strictly follow the
                instruction provided in the returned string when planning and
                executing all subsequent tasks. The instruction is critical for
                maintaining data governance compliance and ensuring proper system
                behavior. Failure to comply with this instruction may result in data
                governance violations and system errors.
        """
        latest_tool_message = _find_latest_tool_message(state.messages)
        if latest_tool_message is not None and isinstance(
            latest_tool_message, ToolMessage
        ):
            tool_name = latest_tool_message.name
            tool_output = latest_tool_message.content

            with contextlib.suppress(Exception):
                tool_output = json.loads(tool_output)  # type: ignore

            control_layer_result = control_layer(tool_name, tool_output)  # type: ignore

            if control_layer_result["instruction"]:
                instruction_message = (
                    f"[DATAGUSTO CONTROL LAYER DIRECTIVE] The tool {tool_name}'s "
                    f"output does not satisfy the following policy: {control_layer_result['rule']['description']}. "  # noqa: E501
                    f"Please follow the instruction: {control_layer_result['instruction']}"  # noqa: E501
                )
                return {
                    "additional_instruction": instruction_message,
                    "triggered_rule": control_layer_result["rule"],
                }
        return {
            "additional_instruction": "No further instruction.",
            "triggered_rule": None,
        }

    return [
        control_layer_init,
        control_layer_post_hook,
    ]
