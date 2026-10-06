"""The graph's behavior, with a scripted model so the tests are fast and repeatable."""

from __future__ import annotations

from typing import Any, List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import Field

from order_agent import data
from order_agent.graph import build_graph, turn
from order_agent.text import text_of


class ScriptedModel(BaseChatModel):
    """Returns the scripted replies in order, whoever asks (main agent or sub-agent)."""

    script: List[AIMessage]
    seen: List[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, call_id="c1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def run(script, max_rounds=6):
    model = ScriptedModel(script=script)
    app = build_graph(model, checkpointer=MemorySaver(), max_rounds=max_rounds)
    return model, app, {"configurable": {"thread_id": "t"}}


@pytest.fixture(autouse=True)
def fresh_orders():
    data.reset()


def test_reads_then_answers():
    model, app, cfg = run([
        call("search_orders", {"clinic": "bayview"}),
        AIMessage("Bayview has SO-1001 (processing) and SO-1002 (shipped)."),
    ])
    out = app.invoke(turn("What does Bayview have?"), cfg)
    assert out["tools_used"] == ["search_orders"]
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "SO-1001" in tool_msg.content and "SO-1002" in tool_msg.content
    assert text_of(out["messages"][-1]).startswith("Bayview has")
    assert not model.script


def test_write_waits_for_approval_then_runs():
    _, app, cfg = run([
        call("cancel_order", {"order_id": "SO-1001", "reason": "clinic asked"}),
        AIMessage("SO-1001 has been cancelled."),
    ])
    out = app.invoke(turn("Cancel SO-1001"), cfg)
    card = out["__interrupt__"][0].value["cards"][0]
    assert card["tool"] == "cancel_order"
    assert data.ORDERS[0]["status"] == "processing"  # nothing ran before approval

    out = app.invoke(Command(resume={"approved": [card["id"]]}), cfg)
    assert data.ORDERS[0]["status"] == "cancelled"
    assert out["actions_done"] == ["cancel_order"]
    assert "[check]" not in str(out["messages"])


def test_declined_write_does_not_run():
    _, app, cfg = run([
        call("cancel_order", {"order_id": "SO-1001", "reason": "clinic asked"}),
        AIMessage("Not cancelled: staff declined. SO-1001 is still processing."),
    ])
    app.invoke(turn("Cancel SO-1001"), cfg)
    out = app.invoke(Command(resume={"approved": []}), cfg)
    assert data.ORDERS[0]["status"] == "processing"
    assert "declined" in [m for m in out["messages"] if isinstance(m, ToolMessage)][-1].content


def test_check_sends_back_unsourced_figures():
    model, app, cfg = run([
        AIMessage("You have about 5 orders worth $3,000."),
        call("order_stats", {}),
        AIMessage("5 orders worth $5,198.00 in total."),
    ])
    out = app.invoke(turn("How are orders looking?"), cfg)
    corrections = [m for m in out["messages"] if isinstance(m, HumanMessage) and "[check]" in m.content]
    assert len(corrections) == 1
    assert out["tools_used"] == ["order_stats"]
    assert text_of(out["messages"][-1]) == "5 orders worth $5,198.00 in total."


def test_check_catches_claimed_change():
    _, app, cfg = run([
        AIMessage("I've cancelled SO-1003."),
        AIMessage("SO-1003 has not been changed yet; I need approval to cancel it."),
    ])
    out = app.invoke(turn("Cancel SO-1003"), cfg)
    assert any("[check]" in str(m.content) for m in out["messages"] if isinstance(m, HumanMessage))
    assert data.ORDERS[2]["status"] == "processing"


def test_check_corrects_only_once():
    _, app, cfg = run([AIMessage("Around 9 orders."), AIMessage("Still around 9 orders.")])
    out = app.invoke(turn("How many orders?"), cfg)
    assert text_of(out["messages"][-1]) == "Still around 9 orders."


def test_round_limit_forces_an_answer():
    _, app, cfg = run([
        call("search_orders", {}, "c1"),
        call("order_stats", {}, "c2"),
        AIMessage("Partial: I looked up the orders but stopped before the product check."),
    ], max_rounds=2)
    out = app.invoke(turn("Audit everything"), cfg)
    assert out["tools_used"] == ["search_orders", "order_stats"]
    assert text_of(out["messages"][-1]).startswith("Partial")


def test_subagent_returns_only_its_summary():
    _, app, cfg = run([
        call("research_clinic", {"clinic": "Elm Street", "question": "anything stuck?"}, "p1"),
        call("search_orders", {"clinic": "Elm Street"}, "s1"),           # sub-agent
        AIMessage("SO-1003 is open but IMP-4511 is out of stock."),      # sub-agent summary
        AIMessage("Elm Street: SO-1003 is waiting on IMP-4511, which is out of stock."),
    ])
    out = app.invoke(turn("Anything stuck for Elm Street?"), cfg)
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["research_clinic"]  # sub-agent's own calls stay inside it
    assert tool_msgs[0].content == "SO-1003 is open but IMP-4511 is out of stock."


def test_conversation_is_kept_per_thread():
    model, app, cfg = run([AIMessage("Hello."), AIMessage("You said hi.")])
    app.invoke(turn("hi"), cfg)
    app.invoke(turn("what did I say?"), cfg)
    last_prompt = model.seen[-1]
    assert [m.content for m in last_prompt if isinstance(m, HumanMessage)] == ["hi", "what did I say?"]


def test_check_catches_garbled_order_ids():
    _, app, cfg = run([
        call("search_orders", {"clinic": "Elm Street"}),
        AIMessage("Orders 101 and 103 are still processing."),
        AIMessage("SO-1003 is still processing; SO-1004 is completed."),
    ])
    out = app.invoke(turn("Anything open for Elm Street?"), cfg)
    check = [m.content for m in out["messages"] if isinstance(m, HumanMessage) and "[check]" in m.content]
    assert check and "SO-101" in check[0] and "SO-103" in check[0]
    assert text_of(out["messages"][-1]).startswith("SO-1003")


def test_ids_from_earlier_turns_need_a_fresh_lookup():
    _, app, cfg = run([
        call("search_orders", {"clinic": "Bayview"}),
        AIMessage("Bayview: SO-1001 and SO-1002."),
        AIMessage("SO-1001 is the open one."),
        AIMessage("SO-1001 is the open one."),
    ])
    app.invoke(turn("Bayview orders?"), cfg)
    out = app.invoke(turn("Which is open?"), cfg)
    assert any("[check]" in str(m.content) for m in out["messages"][-3:])


def test_ids_with_typographic_hyphens_are_checked():
    _, app, cfg = run([
        call("search_orders", {"clinic": "Elm Street"}),
        AIMessage("SO‑1009 is processing."),
        AIMessage("SO‑1003 is processing."),
    ])
    out = app.invoke(turn("Elm Street?"), cfg)
    check = [m.content for m in out["messages"] if isinstance(m, HumanMessage) and "[check]" in m.content]
    assert check and "SO-1009" in check[0]
    assert text_of(out["messages"][-1]) == "SO‑1003 is processing."


def test_failed_write_does_not_count_as_done():
    _, app, cfg = run([
        call("cancel_order", {"order_id": "SO-1002", "reason": "late"}),
        AIMessage("SO-1002 has been cancelled."),
        AIMessage("SO-1002 already shipped, so it can't be cancelled."),
    ])
    out = app.invoke(turn("Cancel SO-1002"), cfg)
    out = app.invoke(Command(resume={"approved": [out["__interrupt__"][0].value["cards"][0]["id"]]}), cfg)
    assert out["actions_done"] == []
    assert data.ORDERS[1]["status"] == "shipped"
    assert any("[check]" in str(m.content) for m in out["messages"] if isinstance(m, HumanMessage))


def test_true_statements_are_not_claims_of_a_change():
    from order_agent.graph import _claims_change
    assert not _claims_change("SO-1002 has been shipped, so it can't be cancelled.")
    assert not _claims_change("No orders have been cancelled.")
    assert not _claims_change("SO-1003 has not been changed yet.")
    assert _claims_change("Done. SO-1003 has been cancelled.")
    assert _claims_change("I've updated SO-1001 to shipped.")


def test_round_limit_holds_even_if_the_model_keeps_calling_tools():
    _, app, cfg = run([
        call("search_orders", {}, "c1"),
        call("order_stats", {}, "c2"),
        call("search_orders", {}, "c3"),   # still asking for tools after the limit
    ], max_rounds=2)
    out = app.invoke(turn("Audit everything"), cfg)
    last = out["messages"][-1]
    assert not last.tool_calls and "step limit" in text_of(last)


def test_small_numbers_are_not_order_ids():
    from order_agent.graph import _unsourced_ids
    assert _unsourced_ids("Orders 1 and 2 of 2024 were fine.", [HumanMessage("hi")]) == []


@pytest.mark.parametrize("query,skus", [
    ("4.5 x 11.5 mm implant", ["IMP-4511"]),        # issue #1: word order and sizes
    ("implant 4.5x11.5", ["IMP-4511"]),
    ("IMP-4511", ["IMP-4511"]),
    ("abutment", ["ABT-STR", "ABT-ANG"]),
    ("healing cap", ["HEA-45"]),
    ("titanium crown", []),
])
def test_product_lookup_matches_words_and_sizes(query, skus):
    from order_agent.tools import product_lookup
    assert [p["sku"] for p in product_lookup.invoke({"query": query})] == skus
