# agent-control-layer

`agent-control-layer` is a library designed to add a policy-based control layer to LLM agent frameworks like LangChain and LangGraph. This allows for more granular control over agent behavior, enabling restrictions on specific tool usage and detection of prompt modifications.

## Overview

Have you ever experienced your LLM agent calling a tool, only to receive fewer or unexpected results, forcing you back into endless, fragile prompt tweaking? Or perhaps you've found yourself hard-coding every possible scenario, leading to scattered and unmanageable control logic?

`agent-control-layer` addresses these common pain points by providing a structured, configurable way to manage agent behavior. Instead of relying on unpredictable prompt engineering or brittle hard-coded rules, this library allows you to define behavior rules in YAML. When a specified tool runs and meets a trigger condition (e.g., `len(tool_output) < 3` for a `web_search` tool), additional instructions are automatically injected into the agent's thought process.

This approach offers significant advantages:
- **Maintainable**: All control logic resides in a centralized, easy-to-manage location.
- **Testable**: Rules are defined as structured configuration, making them testable like code.
- **Collaborative**: Non-technical team members can understand and even modify behavior rules.
- **Debuggable**: Provides a clear audit trail of what triggered when, simplifying debugging.

Key Features:
- **Contract-Based Policy Engine**: Define per-tool contracts in YAML (`.dg_acl/*.yaml`) with prioritized rules that describe when and how the agent should adapt its behaviour.
- **Dynamic Instruction Injection**: When a rule's trigger condition is met, the library automatically injects additional instructions into the agent's reasoning loop—no manual prompt fiddling required.
- **Safe Condition Evaluation**: Trigger conditions are evaluated inside a restricted sandbox to prevent unsafe code execution.
- **Seamless LangGraph Integration**: Drop-in helper function `build_control_layer_tools` exposes `control_layer_init` and `control_layer_post_hook` tools for effortless use in LangGraph workflows.

## Examples

A fully working sample project that demonstrates `agent-control-layer` in a real LangGraph agent can be found in [`examples/langgraph-react-agent`](./examples/langgraph-react-agent).
If you prefer to learn by example, we recommend running this project first and exploring its code before integrating the library into your own application.

## Installation

`agent-control-layer` can be installed using `uv` or `pip`.

```bash
uv add -U agent-control-layer
# or
pip install -U agent-control-layer
```

## Basic Usage

Here's an example of integrating `agent-control-layer` into an existing LangGraph agent.

### Integrating with a LangGraph Agent

To integrate `agent-control-layer` with your LangGraph agent, open your agent's graph definition file (e.g., `src/react_agent/graph.py` if you're using a template like `langgraph-react-agent`). Import `build_control_layer_tools` from `agent_control_layer.langgraph` and add its output to your agent's existing tool list.

```python
# Example: src/react_agent/graph.py

from agent_control_layer.langgraph import build_control_layer_tools

# ... existing TOOLS definition ...

TOOLS = TOOLS + build_control_layer_tools(State)

# ... rest of the code ...
```

The `build_control_layer_tools` function returns control tools provided by `agent-control-layer`. These tools are automatically used by the agent to evaluate and enforce policies.

### Creating Policy Files

`agent-control-layer` looks for contract files in a directory named `.dg_acl` located at your project root:

```
my-agent-project/
├── .dg_acl/
│   ├── contract_file_1.yaml
│   ├── contract_file_2.yaml
│   ├── ...
│   └── contract_file_N.yaml
└── src/
    └── ...
```

A **single YAML file describes the contract for one tool** (identified by `tool_name`).
The file must contain the following top-level keys:

| Key | Type | Description |
|-----|------|-------------|
| `tool_name` | `str` | The exact name of the tool to which this contract applies. |
| `description` | `str` | Human-readable summary of what the contract enforces. |
| `rules` | `list[Rule]` | Ordered list of rules. The library sorts them by `priority` (ascending) before evaluation. |

Each `Rule` item supports:

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | `str` | ✓ | Unique identifier for the rule. |
| `description` | `str` | ✓ | What the rule checks / enforces. |
| `trigger_condition` | `str` | ✓ | Python expression evaluated against `tool_output`. Must return `True` to fire. |
| `instruction` | `str` | ✓ | Additional directive injected into the agent when the rule fires. |
| `priority` | `int` | ✓ | Lower numbers are evaluated first. |

Below is the full `search.yaml` from the `examples` directory. It shows two rules that verify the result count and relevance of a hypothetical `search` tool.

```yaml
# .dg_acl/search.yaml

tool_name: "search"
description: "Rules for the search tool"
rules:
  # Rule 1 – result count
  - name: "search_result_count"
    description: "The search tool must return at least 5 results."
    trigger_condition: "len(tool_output['results']) < 5"
    instruction: "Ask the user to approve to use the results for further steps."
    priority: 1

  # Rule 2 – relevance check
  - name: "search_result_relevance"
    description: "The search tool must return results that are relevant to the user's query."
    trigger_condition: "len([r for r in tool_output['results'] if r['score'] >= 0.5]) < 5"
    instruction: "Ask the user to approve to use the results for further steps."
    priority: 2
```

When the `search` tool finishes, `agent-control-layer` evaluates these rules in order of priority. The **first rule whose `trigger_condition` evaluates to `True` wins**, and its `instruction` is injected into the agent via `control_layer_post_hook`.

> 💡 **Tip:** Because contracts are plain YAML, you can version-control them, write unit tests against them, and allow non-developers to propose changes without touching code.
