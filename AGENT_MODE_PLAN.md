# Agent Mode: Tool-Use Research Sessions

## Problem

The current loop is one-shot: LLM generates strategy code → backtest → keep/discard.
The LLM never sees the data. It guesses blindly and iterates on metric feedback alone.

A real quant researcher: explores data → forms hypotheses → tests them → refines → submits.

## Solution

Replace the single `messages.create()` call with an Anthropic **tool-use loop**.
The LLM gets tools to explore data and test strategies within a single iteration.

## Tools

### 1. `explore` — run arbitrary code on the bars, see output

```json
{
  "name": "explore",
  "description": "Run Python code on the bars data. Returns stdout (max 2000 chars). Use to compute statistics, check correlations, profile distributions, test hypotheses.",
  "input_schema": {
    "type": "object",
    "properties": {
      "code": {"type": "string", "description": "Python code to execute. `bars` dict is available. numpy as np, pandas as pd."}
    },
    "required": ["code"]
  }
}
```

Implementation: reuse sandbox subprocess. Instead of calling `strategy(bars)`,
exec the code snippet directly and capture stdout. Same timeout, same env.

### 2. `backtest` — test a strategy and get metrics

```json
{
  "name": "backtest",
  "description": "Write a strategy function and backtest it. Returns Sharpe, drawdown, trades, win rate, exposure.",
  "input_schema": {
    "type": "object",
    "properties": {
      "strategy_code": {"type": "string", "description": "Full Python file defining def strategy(bars) -> positions array."}
    },
    "required": ["strategy_code"]
  }
}
```

Implementation: write strategy_code to a temp file, call existing `run_strategy()`,
then `backtest()`. Return the metrics dict as the tool result. The LLM can call
this multiple times to compare variants.

### 3. `submit` — commit the final strategy

```json
{
  "name": "submit",
  "description": "Submit your best strategy as this iteration's final answer. It will be tortured-tested and kept or discarded.",
  "input_schema": {
    "type": "object",
    "properties": {
      "strategy_code": {"type": "string", "description": "Final strategy code to submit."}
    },
    "required": ["strategy_code"]
  }
}
```

Implementation: write to strategy.py, run full torture suite (noise + deflation +
walkforward), apply keep/discard logic. This ends the tool-use loop.

## Flow

```
Outer loop (same as today):
  for each iteration:
    start tool-use conversation
    while LLM keeps calling tools:
      if explore → sandbox exec, return stdout
      if backtest → sandbox + backtest, return metrics
      if submit → full torture suite, keep/discard, break
    log experiment, update session
```

## Implementation Plan

### File: sandbox.py — add `run_explore()`

New function (~20 lines):
```python
def run_explore(code: str, bars: dict, timeout_seconds: int = 10) -> str:
    """Execute code snippet with bars in scope, return stdout."""
```

Uses a simpler RUNNER_TEMPLATE that exec()s the code and captures print output.
Same subprocess pattern as run_strategy but returns stdout string, not positions.

### File: loop.py — replace `run_iteration()`

Replace the single messages.create() with a tool-use loop (~60 lines):

```python
def run_iteration_agent(client, bars_np, close_returns, best_sharpe, model):
    tools = [EXPLORE_TOOL, BACKTEST_TOOL, SUBMIT_TOOL]
    messages = [{"role": "user", "content": user_msg}]

    while True:
        response = client.messages.create(
            model=model, max_tokens=4096,
            system=build_system_prompt(), tools=tools,
            messages=messages,
        )

        # Process tool calls
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "explore":
                    result = run_explore(block.input["code"], bars_json)
                    # append tool_result to messages
                elif block.name == "backtest":
                    # write temp file, run_strategy, backtest, return metrics
                elif block.name == "submit":
                    # write strategy.py, full torture suite, return
                    return experiment

        if response.stop_reason == "end_turn":
            # LLM finished without submitting — treat as no-op
            break

        # Safety: cap at 20 tool calls per iteration
        if len(messages) > 40:
            break
```

### File: loop.py — keep `run_iteration()` as fallback

Rename current function to `run_iteration_oneshot()`. New `run_iteration()` calls
agent mode by default, falls back to oneshot if tools not supported.

### File: cli.py — add `--agent/--oneshot` flag

Default to agent mode. `--oneshot` for the old behavior.

### System prompt update

Add tool-use instructions:

```
You have three tools:
- explore: run code to analyze the data before writing a strategy
- backtest: test a strategy variant and see its metrics
- submit: commit your best strategy for torture testing

Workflow: explore the data first. Form hypotheses. Test 2-3 strategy variants
with backtest. Submit the best one. Each iteration is a research session.
```

## What Changes

| Component | Change |
|-----------|--------|
| sandbox.py | +20 lines (run_explore) |
| loop.py | +60 lines (agent loop), rename old to oneshot |
| cli.py | +5 lines (--agent/--oneshot flag) |
| system prompt | Updated with tool instructions |
| tests | +3 tests for explore, agent loop |
| **Total** | **~100 lines net** |

## What Doesn't Change

- backtest.py — untouched
- torture.py — untouched
- datasource.py — untouched
- firewall.py — untouched
- strategy_template.py — untouched
- The outer loop structure — untouched
- Session persistence (experiments.jsonl, session.md, git) — untouched

## Token Cost

Each iteration uses more tokens (exploration + multiple backtests + reasoning).
Rough estimate: 3-5x more tokens per iteration, but much higher quality per iteration.
With 5 explore calls + 3 backtests + 1 submit = ~9 tool calls = ~10K tokens per iteration
vs ~2K tokens for one-shot.

This is the perfect use case for the token oracle's end-of-week budget burns.

## Vet Findings (Applied)

### Critical: Anthropic message threading
The tool loop MUST: (1) append the full assistant response to messages,
(2) collect ALL tool_results into a SINGLE user message, (3) then call
create() again. Missing step 1 causes API validation errors.

### Critical: Exploration overfitting
If explore sees the same data as backtest, the LLM can compute exact
statistics and bake them into the strategy. Fix: explore gets first 70%
of bars. Submit torture-tests on 100%.

### High: Hill-climbing via repeated backtest
Cap backtest calls at 5 per iteration. Use random seed for noise test
(not fixed 42). Consider returning qualitative grades instead of exact
Sharpe from backtest tool.

### Medium: User message incompatible with tool use
Current message says "Output ONLY the Python code." Must be rewritten
for agent mode to instruct tool use instead.

### Medium: Context growth
20 tool calls × 2K chars each = ~5K tokens of results in context.
Consider pruning old explore results after 3 turns.

## Risks

1. **LLM doesn't call submit** — handled by end_turn detection + max tool call cap
2. **Explore runs expensive code** — 10s timeout per explore call
3. **Too many tool calls** — explicit counter, cap at 20
4. **Too many backtests** — per-tool cap at 5
5. **Model doesn't support tools** — fallback to oneshot mode
6. **Parallel tool calls** — submit takes precedence if mixed with other tools
