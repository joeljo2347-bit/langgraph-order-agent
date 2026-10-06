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
    r"\b(?:I(?:'ve| have)?\s+(?:just\s+)?(?:cancell?ed|updated|changed|marked)"
    r"|(?:has|have) been\s+(?:cancell?ed|updated|changed|marked))\b",
    re.I,
)
_NEGATED = re.compile(r"\b(?:no|not|never|nothing|cannot|can't|couldn't|wasn't|hasn't|haven't)\b|n't\b", re.I)


def _claims_change(answer: str) -> bool:
    """A sentence saying a change was made. "No orders have been cancelled" and "SO-1002 has been
    shipped, so it can't be cancelled" are not claims of a change."""
    return any(_CLAIMS.search(s) and not _NEGATED.search(s)
               for s in re.split(r"(?<=[.!?])\s+|\n+", answer))


_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})
_IDS = re.compile(r"\bSO-(\d+)\b", re.I)
_BARE_IDS = re.compile(r"\borders?\s+#?(\d{3,}(?:\s*(?:,|and|&)\s*#?\d{3,})*)", re.I)


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



def _problem(state: State) -> Optional[str]:
    """What's wrong with the final answer, if anything (the three checks)."""
    answer = text_of(state["messages"][-1])
    if _FIGURES.search(answer) and not set(state["tools_used"]) & DATA_TOOLS:
        return ("Your answer has order figures but no data tool ran this turn. "
                "Look them up, then answer again.")
    if _claims_change(answer) and not state["actions_done"]:
        return ("Your answer says a change was made, but no approved change ran this turn. "
                "Call the tool for it, or say it has not been done.")
    unsourced = _unsourced_ids(answer, state["messages"])
    if unsourced:
        return (f"Your answer names {', '.join(unsourced)}, which no tool returned this turn. "
                "Use order ids exactly as the tool results give them.")
    return None


def check(state: State) -> Command[Literal["agent", "__end__"]]:
    """Send a failing answer back to the model once; otherwise finish."""
    problem = None if state["corrected"] else _problem(state)
    if problem is None:
        return Command(goto=END)
    return Command(goto="agent", update={"messages": [HumanMessage(f"[check] {problem}")],
                                         "corrected": True})


def after_agent(state: State) -> Literal["tools", "check"]:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check"


def _approved_ids(calls: List[Dict[str, Any]]) -> set:
    """Pause for a person's decision on any writes. The graph stops here and is saved; the caller
    resumes with Command(resume={"approved": [call ids]}). Nothing has run yet."""
    writes = [c for c in calls if c["name"] in WRITE_TOOLS]
    if not writes:
        return set()
    decision = interrupt({"cards": [{"id": c["id"], "tool": c["name"], "args": c["args"]} for c in writes]})
    return set((decision or {}).get("approved", []))


class Nodes:
    """The graph's model-dependent nodes: the agent step and the tool step."""

    def __init__(self, model: BaseChatModel, max_rounds: int):
        tools = READ_TOOLS + ACTION_TOOLS + [make_research_tool(model)]
        self.by_name = {t.name: t for t in tools}
        self.with_tools = model.bind_tools(tools)
        self.max_rounds = max_rounds

    def agent(self, state: State) -> Dict[str, Any]:
        messages = [SystemMessage(SYSTEM)] + state["messages"]
        if state["rounds"] < self.max_rounds:
            return {"messages": [self.with_tools.invoke(messages)], "rounds": state["rounds"] + 1}
        # Out of rounds. Keep the tools bound (some providers reject a history with tool calls but
        # no tools), ask for an answer, and drop any tool call that still comes back.
        note = HumanMessage("[limit] No more tool calls this turn. Answer with what you have "
                            "and say what you could not finish.")
        reply = self.with_tools.invoke(messages + [note])
        if getattr(reply, "tool_calls", None):
            reply = AIMessage(text_of(reply) or "I couldn't finish this within the step limit.")
        return {"messages": [reply], "rounds": state["rounds"] + 1}

    def _run_one(self, call: Dict[str, Any], approved: set) -> tuple:
        """(result text, whether it ran) for one tool call."""
        name, call_id = call["name"], call["id"]
        if name not in self.by_name:
            return f"Unknown tool {name}.", False
        if name in WRITE_TOOLS and call_id not in approved:
            return "Not done: staff declined this change.", False
        if name in WRITE_TOOLS and call_id in data.DONE_ACTIONS:
            return "Already done.", False
        try:
            result = self.by_name[name].invoke(call["args"])
        except Exception as exc:  # noqa: BLE001 - the model sees the error and can recover
            return f"Not done: {exc}", False
        if name in WRITE_TOOLS:
            data.DONE_ACTIONS.add(call_id)
        return (result if isinstance(result, str) else json.dumps(result)), True

    def tools(self, state: State) -> Dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        approved = _approved_ids(calls)
        used, done, out = list(state["tools_used"]), list(state["actions_done"]), []
        for call in calls:
            content, ran = self._run_one(call, approved)
            if ran:
                used.append(call["name"])
                done += [call["name"]] if call["name"] in WRITE_TOOLS else []
            out.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": out, "tools_used": used, "actions_done": done}


def build_graph(model: BaseChatModel, checkpointer: Optional[Any] = None, max_rounds: int = MAX_ROUNDS):
    nodes = Nodes(model, max_rounds)
    graph = StateGraph(State)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", nodes.tools)
    graph.add_node("check", check)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
