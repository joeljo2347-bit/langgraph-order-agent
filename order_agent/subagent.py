"""research_clinic: a sub-agent the main agent calls like a tool.

It runs as its own graph with a fresh context and only two read tools, and hands back a short
summary. The main conversation grows by that paragraph, not by the sub-agent's tool calls.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent

from order_agent.text import text_of
from order_agent.tools import product_lookup, search_orders

SUMMARY_LIMIT = 800

_PROMPT = (
    "You research one clinic's orders for a colleague. Use the tools, then reply with a summary "
    "of at most five lines: what they order, what is open, anything that needs attention "
    "(out-of-stock items, long-open orders). Use order ids. No preamble."
)


def make_research_tool(model: BaseChatModel) -> BaseTool:
    researcher = create_react_agent(model, [search_orders, product_lookup], prompt=_PROMPT)

    @tool
    def research_clinic(clinic: str, question: str) -> str:
        """Research one clinic's order history and open issues; returns a short summary."""
        result = researcher.invoke(
            {"messages": [HumanMessage(f"Clinic: {clinic}\nQuestion: {question}")]},
            {"recursion_limit": 12},
        )
        return text_of(result["messages"][-1])[:SUMMARY_LIMIT]

    return research_clinic
