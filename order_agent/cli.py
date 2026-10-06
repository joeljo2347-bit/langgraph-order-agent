"""Chat with the order agent in the terminal.

    python -m order_agent.cli                         # local gpt-oss:20b through Ollama
    python -m order_agent.cli --model anthropic:claude-sonnet-5-5

Writes stop for your approval. The conversation is checkpointed per thread, so follow-ups work.
"""

from __future__ import annotations

import argparse
import json
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from order_agent import models
from order_agent.graph import build_graph, turn
from order_agent.text import text_of


def approve(cards):
    approved = []
    for card in cards:
        args = json.dumps(card["args"])
        if input(f"  approve {card['tool']} {args}? [y/N] ").strip().lower() == "y":
            approved.append(card["id"])
    return {"approved": approved}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=models.DEFAULT)
    args = parser.parse_args()

    app = build_graph(models.load(args.model), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"Order agent on {args.model}. Ctrl-D to quit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if not text:
            continue
        result = app.invoke(turn(text), config)
        while "__interrupt__" in result:
            cards = result["__interrupt__"][0].value["cards"]
            result = app.invoke(Command(resume=approve(cards)), config)
        if result["tools_used"]:
            print(f"  · tools: {', '.join(result['tools_used'])}")
        print(f"agent> {text_of(result['messages'][-1])}")


if __name__ == "__main__":
    main()
