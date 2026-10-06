"""HTTP API: how another system (a staff dashboard, a chat widget, a ticketing tool) uses the agent.

    POST /threads/{id}/messages   {"text": "..."}        -> reply, or approval cards
    POST /threads/{id}/approvals  {"approved": [ids]}    -> resumes the paused run
    GET  /threads/{id}                                   -> the conversation so far
    GET  /health

A run that reaches a write returns status "needs_approval" with one card per change. Nothing has
changed yet; the caller shows the cards to a person and posts the decision. The thread's state is
in the checkpointer, so the approval can come minutes later, from another request.

    uvicorn order_agent.api:app --port 8000
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from order_agent import models
from order_agent.graph import build_graph, turn
from order_agent.text import text_of


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ApprovalIn(BaseModel):
    approved: List[str] = Field(default_factory=list, description="Ids of the cards to run; the rest are declined.")


class Card(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]


class RunOut(BaseModel):
    status: str  # "done" or "needs_approval"
    reply: Optional[str] = None
    cards: List[Card] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    corrected: bool = False


router = APIRouter()


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _pending(graph, thread_id: str) -> bool:
    """Waiting on an approval. A run that crashed also leaves `next` set, but no interrupt,
    and must not lock the thread."""
    return any(task.interrupts for task in graph.get_state(_config(thread_id)).tasks)


def _out(result: Dict[str, Any]) -> RunOut:
    if "__interrupt__" in result:
        return RunOut(status="needs_approval",
                      cards=[Card(**c) for c in result["__interrupt__"][0].value["cards"]])
    return RunOut(status="done", reply=text_of(result["messages"][-1]),
                  tools_used=result["tools_used"], corrected=result["corrected"])


def _visible(m) -> Optional[Dict[str, Any]]:
    """A message as the thread view shows it, or None for internal ones."""
    if isinstance(m, HumanMessage) and not text_of(m).startswith(("[check]", "[limit]")):
        return {"role": "user", "text": text_of(m)}
    if isinstance(m, AIMessage) and not m.tool_calls:
        return {"role": "agent", "text": text_of(m)}
    if isinstance(m, ToolMessage):
        return {"role": "tool", "name": m.name, "text": text_of(m)[:500]}
    return None


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/threads/{thread_id}/messages", response_model=RunOut)
def post_message(thread_id: str, body: MessageIn, request: Request) -> RunOut:
    graph = request.app.state.graph
    if _pending(graph, thread_id):
        raise HTTPException(409, "This thread is waiting for an approval decision.")
    return _out(graph.invoke(turn(body.text), _config(thread_id)))


@router.post("/threads/{thread_id}/approvals", response_model=RunOut)
def post_approval(thread_id: str, body: ApprovalIn, request: Request) -> RunOut:
    graph = request.app.state.graph
    if not _pending(graph, thread_id):
        raise HTTPException(409, "Nothing in this thread is waiting for approval.")
    return _out(graph.invoke(Command(resume={"approved": body.approved}), _config(thread_id)))


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str, request: Request) -> Dict[str, Any]:
    graph = request.app.state.graph
    state = graph.get_state(_config(thread_id))
    if not state.values:
        raise HTTPException(404, "No such thread.")
    messages = [v for v in map(_visible, state.values["messages"]) if v]
    return {"thread_id": thread_id, "waiting_for_approval": _pending(graph, thread_id), "messages": messages}


def create_app(model: Optional[BaseChatModel] = None) -> FastAPI:
    model = model or models.load(os.environ.get("ORDER_AGENT_MODEL", models.DEFAULT))
    app = FastAPI(title="Order agent", version="1.0")
    app.state.graph = build_graph(model, checkpointer=MemorySaver())
    app.include_router(router)
    return app


_app = None


def __getattr__(name: str):  # `uvicorn order_agent.api:app` builds the app once, on first use
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
