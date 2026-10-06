# LangGraph order agent

[![tests](https://github.com/joeljo2347-bit/langgraph-order-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/joeljo2347-bit/langgraph-order-agent/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An AI agent for a dental implant supplier's staff. It answers questions about orders from live data,
hands broad questions to a sub-agent, and **never changes an order until a person approves it**.
Built with LangGraph, served over HTTP, runs on a local model or on Claude.

```console
$ curl -X POST localhost:8000/threads/demo/messages -H 'Content-Type: application/json' \
    -d '{"text": "Cancel SO-1005, the clinic changed their plan."}'
{"status": "needs_approval", "reply": null,
 "cards": [{"id": "3f55…", "tool": "cancel_order", "args": {"order_id": "SO-1005", "reason": "Clinic changed their plan"}}],
 "tools_used": [], "corrected": false}

$ curl -X POST localhost:8000/threads/demo/approvals -H 'Content-Type: application/json' \
    -d '{"approved": ["3f55…"]}'
{"status": "done", "reply": "Order SO-1005 has been cancelled.", "cards": [],
 "tools_used": ["cancel_order"], "corrected": false}
```
<sub>Real output from gpt-oss:20b running locally (ids shortened).</sub>

## Where this comes from

At AMII, a dental implant company, I build and run Noah, the company's AI assistant and ordering
platform. Staff use it to look things up and get work done on orders, and nothing it proposes to
change happens without a staff member's confirmation. This repo rebuilds the core of that agent
from scratch in LangGraph, on made-up data; AMII's code and data stay private.

## How it works

```mermaid
graph TD;
	start([request]) --> agent;
	agent -. tool calls .-> tools;
	tools -- "write? pause for approval" --> agent;
	agent -. answer .-> check;
	check -. "doesn't hold up: correct once" .-> agent;
	check -. ok .-> done([reply]);
```

| Piece | What it does | Where |
|---|---|---|
| Tool loop | The model calls order, stats and catalog tools until it can answer; capped at 6 rounds per turn, then it must answer with what it has | `order_agent/graph.py` |
| Human approval | Any write (cancel, status change) pauses the graph with `interrupt()` and returns one card per change. Nothing has run yet. The run resumes from the checkpoint when a person approves or declines, even minutes later from another request | `graph.py` `run_tools` |
| Sub-agent | `research_clinic` is its own agent with two read tools and a fresh context. The main agent sees only its five-line summary, not its tool trace | `order_agent/subagent.py` |
| Answer checks | Three rules on every final answer: figures need a data tool behind them; "I've cancelled…" needs a change that actually ran; every order id must come from this turn. A failing answer goes back to the model once | `graph.py` `check` |
| HTTP API | Messages, approvals, thread history; a thread waiting for approval refuses new messages (409) | `order_agent/api.py` |
| Any model | `ollama:<model>` locally or `anthropic:<model>`; the graph doesn't change | `order_agent/models.py` |

## Evals on real models

<!-- AGENT-EVALS -->

## Tests

`pytest` runs 21 tests with a scripted stand-in model, no LLM needed: approve, decline, a write
that fails, each answer check, the round limit, sub-agent isolation, follow-up memory, and the
HTTP API end to end. CI runs them on Python 3.9 and 3.12 and builds and starts the Docker image.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pytest -q                                       # no model needed
ollama pull gpt-oss:20b
.venv/bin/python -m order_agent.cli                       # chat in the terminal
.venv/bin/uvicorn order_agent.api:app --port 8000         # or serve the API (docs at /docs)
ANTHROPIC_API_KEY=... .venv/bin/python -m order_agent.cli --model anthropic:claude-sonnet-5-5
```

Or with Docker, the API plus Ollama in containers:

```bash
docker compose up -d && docker compose exec ollama ollama pull gpt-oss:20b
```

The image alone, with Ollama on the host: `docker run -p 8000:8000 order-agent` on Docker Desktop;
on Linux add `--add-host=host.docker.internal:host-gateway` and start Ollama with `OLLAMA_HOST=0.0.0.0`.

## Decisions and tradeoffs

- **Approval lives in the graph, not the tool.** A tool can't be talked into running early, and
  declined changes come back to the model as "Not done", so it can say so honestly.
- **Checks in code, not in the prompt.** Prompts reduce mistakes; a check after the answer catches
  the ones left. Each check sends an answer back at most once, so a stubborn model can't loop.
- **The sub-agent returns a summary, not its trace.** The main context grows by a paragraph per
  research question instead of every tool result.
- **Failed writes raise.** A cancel that can't happen (already shipped) doesn't count as done, so
  "I've cancelled it" is caught.
- **In-memory state.** The checkpointer and order book are in memory to keep the demo small; a
  deployment would use a database checkpointer and the real order system's API.

## Limits

- Made-up data: five orders, five products.
- The checks are pattern-based. They catch invented figures, false claims of a change and
  garbled order ids, not every wrong statement: a small model once called an out-of-stock part
  "in stock", which no check here catches.
