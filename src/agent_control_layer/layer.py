import warnings
from typing import Any, Union

from agent_control_layer.config import _config

SAFE_GLOBALS = {
    "len": len,
    "all": all,
    "any": any,
    "isinstance": isinstance,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
}


def _is_rule_triggered(
    rule: dict[str, Any], tool_output: Union[dict, list, str]
) -> bool:
    """Check if the rule is triggered."""
    expression = rule.get("trigger_condition")

    safe_locals = {"tool_output": tool_output}

    if not isinstance(expression, str) or not expression.strip():
        warnings.warn(
            (
                "Error evaluating expression: trigger_condition must be a non-empty "
                f"string, but got {type(expression)}"
            ),
            stacklevel=2,
        )
        return False

    restricted_globals = {"__builtins__": {}, **SAFE_GLOBALS}

    try:
        return eval(expression, restricted_globals, safe_locals)
    except Exception as e:
        warnings.warn(f"Error evaluating expression: {e}", stacklevel=2)
        return False


def _evaluate_contract(
    contract: dict[str, Any], tool_output: Union[dict, list, str]
) -> dict[str, Any]:
    """Evaluate the contract for a tool."""
    instruction = None
    for rule in contract.get("rules", []):
        if _is_rule_triggered(rule, tool_output):
            instruction = rule.get("instruction")
            return {"instruction": instruction, "rule": rule}

    return {"instruction": None, "rule": None}


def control_layer(
    tool_name: str, tool_output: Union[dict, list, str]
) -> dict[str, Any]:
    """Control layer for the agent."""
    contract = _config.get(tool_name)
    if not contract:
        return {"instruction": None, "rule": None}

    result = _evaluate_contract(contract, tool_output)
    return result
