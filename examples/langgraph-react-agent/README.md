# Reproducing the LangGraph ReAct Agent Sample with agent-control-layer

This guide shows you how to set up the ReAct agent sample, integrate `agent-control-layer`, and run everything locally.

> **Reference implementation:** A fully generated project that follows these exact steps already lives at `examples/langgraph-react-agent/react-agent-python/`. Browse that folder if you want to inspect the final file layout, configuration, or test suite in more detail.

## 1. Prerequisites

* Python **3.10+** (tested with 3.11)
* [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
* Shell that can read your `.env` file (e.g. via `direnv` or manual export)

```bash
# install uv – skip if you already have it
pip install --upgrade uv
```

## 2. Scaffold the base ReAct agent

Install the LangGraph CLI and generate the template project:

```bash
pip install --upgrade langgraph-cli
langgraph new react-agent-python --template react-agent-python
```

This creates a folder called `react-agent-python/` containing the code from the [react-agent-python](https://github.com/langchain-ai/react-agent) template.

## 3. Install `agent-control-layer`

Inside the newly created project folder, add the library:

```bash
cd react-agent
uv add agent-control-layer  # or: pip install agent-control-layer
```

## 4. Configure API keys

Copy the example file and add your credentials (you can skip keys for providers you do not use):

```bash
cp .env.example .env
```

Minimum required keys:

```
# .env
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
# or, if you prefer OpenAI
OPENAI_API_KEY=...
```

## 5. Add a policy contract

Create a directory named `.dg_acl` at the project root and add a YAML file per tool you want to guard. For the built-in `search` tool you can start with:

```yaml
# .dg_acl/search.yaml
tool_name: "search"
description: "Rules for the search tool"
rules:
  - name: "result_count"
    description: "Require at least five results"
    trigger_condition: "len(tool_output['results']) < 5"
    instruction: "Ask the user whether to continue with fewer results or refine the query."
    priority: 1
```

## 6. Wire `agent-control-layer` into the graph

Open `src/react_agent/graph.py` and add two small changes:

```diff
-from react_agent.tools import TOOLS
+# existing imports…
+from agent_control_layer.langgraph import build_control_layer_tools
+
+from react_agent.tools import TOOLS
+
+TOOLS = TOOLS + build_control_layer_tools(State)
```

Nothing else needs to change – the helper returns two invisible tools (`control_layer_init` and `control_layer_post_hook`) that are automatically called at the correct time.

## 7. Start the agent in LangGraph Studio

```bash
langgraph dev
```

Studio hot-reloads your code; every conversation becomes a new graph run that you can inspect visually.

Alternatively, invoke the graph directly:

```python
from react_agent.graph import graph, State

graph.invoke({"input": "What's the tallest mountain in Switzerland?"})
```
