"""The scenario runner, driven by the scripted model."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from evals.run import run_scenario
from order_agent.graph import build_graph
from tests.test_graph import ScriptedModel, call


def test_a_declined_cancel_is_recorded_and_checked():
    app = build_graph(ScriptedModel(script=[
        call("cancel_order", {"order_id": "SO-1005", "reason": "x"}),
        AIMessage("SO-1005 was not cancelled: staff declined."),
    ]), checkpointer=MemorySaver())
    scenario = {"id": "s", "turns": ["Cancel SO-1005"], "approve": False, "needs_approval": True,
                "final_status": {"SO-1005": "processing"}, "contains": ["not|declin"]}
    out = run_scenario(app, scenario, "t")
    assert out["passed"], out["failures"]
    assert out["approval_decisions"] == [{"tool": "cancel_order", "args": {"order_id": "SO-1005", "reason": "x"},
                                          "staff_decision": "declined"}]
    assert [t["role"] for t in out["transcript"]] == ["staff", "assistant-tool-call", "tool-result", "assistant"]
    assert out["orders_before"] == out["orders_after"]


def test_failures_are_named():
    app = build_graph(ScriptedModel(script=[AIMessage("Done.")]), checkpointer=MemorySaver())
    scenario = {"id": "s", "turns": ["Mark SO-1001 shipped"], "approve": True, "needs_approval": True,
                "final_status": {"SO-1001": "shipped"}}
    out = run_scenario(app, scenario, "t")
    assert out["failures"] == ["no approval was asked for", "SO-1001 is processing, expected shipped"]
