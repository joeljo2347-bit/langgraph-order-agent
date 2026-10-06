"""The agent as a LangGraph state machine.

    START -> agent -> tools -> agent -> ... -> check -> END
                        |                        |
                (pauses for approval      (sends the answer back
                 before any write)         once if it doesn't hold up)

- agent:  the model reads the conversation and either calls tools or answers. After MAX_ROUNDS
          tool rounds in one turn it must answer with what it has.
- tools:  runs the calls. Writes (cancel, status change) pause the graph with interrupt(); a
          person approves or declines each one and the graph resumes from the checkpoint.
- check:  three rules on the finished answer: figures need a data tool behind them, "I've
          cancelled/updated..." needs an approved write this turn, and every order id it names
          must be in this turn's tool results. A failing answer goes back to the model once with
          a correction.
- research_clinic is a sub-agent (its own graph, narrow tools, fresh context); the parent only
  sees its short summary, not its tool trace.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from order_agent import data
from order_agent.subagent import make_research_tool
from order_agent.text import text_of
from order_agent.tools import ACTION_TOOLS, DATA_TOOLS, READ_TOOLS, WRITE_TOOLS

MAX_ROUNDS = 6

SYSTEM = (
    "You help the staff of a dental parts supplier with orders.\n"
    "- Order facts, counts and money figures come only from your tools. Never estimate them.\n"
    "- For a broad question about one clinic (history, what they usually buy, anything odd), "
    "call research_clinic once instead of many lookups.\n"
    "- To cancel an order or change its status, call the tool. A staff member approves it before "
    "it runs. Never say a change is done unless the tool result says so.\n"
    "- Be brief. Use order ids."
)

_FIGURES = re.compile(r"\$\s?\d|\b\d+\s+(?:orders?|units?)\b", re.I)
_CLAIMS = re.compile(
    r"\b(?:I(?:'ve| have)?\s+(?:cancell?ed|updated|changed|marked|shipped)"
    r"|(?:has|have) been\s+(?:cancell?ed|updated|changed|marked|shipped))\b",
    re.I,
)
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})
_IDS = re.compile(r"\bSO-(\d+)\b", re.I)
_BARE_IDS = re.compile(r"\borders?\s+#?(\d+(?:\s*(?:,|and|&)\s*#?\d+)*)", re.I)


def _unsourced_ids(answer: str, messages: List[AnyMessage]) -> List[str]:
    """Order ids in the answer that neither the user, a tool call nor a tool result gave this
    turn ("orders 101 and 103" counts)."""
    answer = answer.translate(_DASHES)
    seen = ""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            seen += text_of(message)
        elif isinstance(message, AIMessage):
            seen += json.dumps([c["args"] for c in message.tool_calls])
        elif isinstance(message, HumanMessage) and not text_of(message).startswith("[check]"):
            seen += text_of(message)
            break
    cited = set(_IDS.findall(answer))
    for group in _BARE_IDS.findall(answer):
        cited.update(re.findall(r"\d+", group))
    seen = seen.translate(_DASHES).upper()
    return sorted(f"SO-{n}" for n in cited if f"SO-{n}" not in seen)


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    rounds: int  # tool rounds this turn
    corrected: bool  # the check already sent this turn's answer back once
    tools_used: List[str]  # tools that ran this turn
    actions_done: List[str]  # approved writes that ran this turn


def turn(text: str) -> Dict[str, Any]:
    """Input for one user turn; resets the per-turn counters."""
    return {"messages": [HumanMessage(text)], "rounds": 0, "corrected": False,
            "tools_used": [], "actions_done": []}



def build_graph(model: BaseChatModel, checkpointer: Optional[Any] = None, max_rounds: int = MAX_ROUNDS):
    tools = READ_TOOLS + ACTION_TOOLS + [make_research_tool(model)]
    by_name = {t.name: t for t in tools}
    with_tools = model.bind_tools(tools)

    def agent(state: State) -> Dict[str, Any]:
        messages = [SystemMessage(SYSTEM)] + state["messages"]
        if state["rounds"] >= max_rounds:
            note = HumanMessage("[limit] No more tool calls this turn. Answer with what you have "
                                "and say what you could not finish.")
            reply = model.invoke(messages + [note])
        else:
            reply = with_tools.invoke(messages)
        return {"messages": [reply], "rounds": state["rounds"] + 1}

    def after_agent(state: State) -> Literal["tools", "check"]:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check"

    def run_tools(state: State) -> Dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        writes = [c for c in calls if c["name"] in WRITE_TOOLS]
        approved: set = set()
        if writes:
            # The graph stops here and is saved; the caller resumes with
            # Command(resume={"approved": [call ids]}). Nothing has run yet.
            decision = interrupt({"cards": [{"id": c["id"], "tool": c["name"], "args": c["args"]}
                                            for c in writes]})
            approved = set((decision or {}).get("approved", []))

        used, done, out = list(state["tools_used"]), list(state["actions_done"]), []
        for call in calls:
            name, call_id = call["name"], call["id"]
            if name not in by_name:
                content = f"Unknown tool {name}."
            elif name in WRITE_TOOLS and call_id not in approved:
                content = "Not done: staff declined this change."
            elif name in WRITE_TOOLS and call_id in data.DONE_ACTIONS:
                content = "Already done."
            else:
                try:
                    result = by_name[name].invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 - the model sees the error and can recover
                    content = f"Not done: {exc}"
                else:
                    content = result if isinstance(result, str) else json.dumps(result)
                    used.append(name)
                    if name in WRITE_TOOLS:
                        data.DONE_ACTIONS.add(call_id)
                        done.append(name)
            out.append(ToolMessage(content, tool_call_id=call_id, name=name))
        return {"messages": out, "tools_used": used, "actions_done": done}

    def check(state: State) -> Command[Literal["agent", "__end__"]]:
        if state["corrected"]:
            return Command(goto=END)
        answer = text_of(state["messages"][-1])
        problem = None
        unsourced = _unsourced_ids(answer, state["messages"])
        if _FIGURES.search(answer) and not set(state["tools_used"]) & DATA_TOOLS:
            problem = ("Your answer has order figures but no data tool ran this turn. "
                       "Look them up, then answer again.")
        elif _CLAIMS.search(answer) and not state["actions_done"]:
            problem = ("Your answer says a change was made, but no approved change ran this turn. "
                       "Call the tool for it, or say it has not been done.")
        elif unsourced:
            problem = (f"Your answer names {', '.join(unsourced)}, which no tool returned this turn. "
                       "Use order ids exactly as the tool results give them.")
        if problem is None:
            return Command(goto=END)
        return Command(goto="agent", update={"messages": [HumanMessage(f"[check] {problem}")],
                                             "corrected": True})

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("check", check)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
