"""The HTTP API, end to end, with the scripted model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from order_agent import data
from order_agent.api import create_app
from tests.test_graph import ScriptedModel, call


@pytest.fixture(autouse=True)
def fresh_orders():
    data.reset()


def client(script):
    return TestClient(create_app(ScriptedModel(script=script)))


def test_question_gets_a_reply():
    c = client([call("order_stats", {}), AIMessage("5 orders, $5,198.00 in total.")])
    r = c.post("/threads/t1/messages", json={"text": "How are orders?"})
    assert r.status_code == 200
    assert r.json() == {"status": "done", "reply": "5 orders, $5,198.00 in total.", "cards": [],
                        "tools_used": ["order_stats"], "corrected": False}


def test_write_returns_cards_then_runs_on_approval():
    c = client([call("cancel_order", {"order_id": "SO-1005", "reason": "plan changed"}, "x1"),
                AIMessage("SO-1005 has been cancelled.")])
    r = c.post("/threads/t2/messages", json={"text": "Cancel SO-1005"}).json()
    assert r["status"] == "needs_approval"
    assert r["cards"] == [{"id": "x1", "tool": "cancel_order",
                           "args": {"order_id": "SO-1005", "reason": "plan changed"}}]
    assert data.ORDERS[4]["status"] == "processing"

    # a new message can't jump the queue while a decision is pending
    assert c.post("/threads/t2/messages", json={"text": "hello?"}).status_code == 409

    r = c.post("/threads/t2/approvals", json={"approved": ["x1"]}).json()
    assert r["status"] == "done" and data.ORDERS[4]["status"] == "cancelled"

    thread = c.get("/threads/t2").json()
    assert thread["waiting_for_approval"] is False
    assert [m["role"] for m in thread["messages"]] == ["user", "tool", "agent"]


def test_approval_without_pending_run_is_rejected():
    c = client([])
    assert c.post("/threads/none/approvals", json={"approved": []}).status_code == 409
    assert c.get("/threads/none").status_code == 404


def test_input_is_validated():
    c = client([])
    assert c.post("/threads/t3/messages", json={"text": ""}).status_code == 422
    assert c.get("/health").json() == {"status": "ok"}


def test_a_crashed_run_does_not_lock_the_thread():
    class Boom(ScriptedModel):
        def _generate(self, *a, **k):
            if not self.script:
                raise RuntimeError("model down")
            return super()._generate(*a, **k)

    c = TestClient(create_app(Boom(script=[])), raise_server_exceptions=False)
    assert c.post("/threads/t4/messages", json={"text": "hi"}).status_code == 500
    r = c.post("/threads/t4/messages", json={"text": "hi again"})
    assert r.status_code != 409
