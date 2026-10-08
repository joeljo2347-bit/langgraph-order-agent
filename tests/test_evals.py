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


class StuckModel(ScriptedModel):
    """Times out on its first call, as a stuck generation does at the client's timeout."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.seen:
            self.seen.append(messages)
            raise TimeoutError("read timed out")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_a_stuck_call_fails_one_scenario_and_the_run_goes_on():
    app = build_graph(StuckModel(script=[AIMessage("SO-1001 is processing.")]), checkpointer=MemorySaver())
    scenario = {"id": "s", "turns": ["Where is SO-1001?"], "contains": ["processing"]}
    stuck = run_scenario(app, scenario, "t1")
    assert not stuck["passed"] and stuck["failures"] == ["model call failed: TimeoutError"]
    assert run_scenario(app, scenario, "t2")["passed"]


def test_a_reply_cut_off_at_the_cap_fails_the_scenario():
    cut = AIMessage("SO-1001 is processing and", response_metadata={"done_reason": "length"})
    app = build_graph(ScriptedModel(script=[cut]), checkpointer=MemorySaver())
    out = run_scenario(app, {"id": "s", "turns": ["Where is SO-1001?"], "contains": ["processing"]}, "t")
    assert out["failures"] == ["1 model call(s) hit the token cap"]
