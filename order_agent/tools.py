"""The tools the agent may call.

Reads run freely. Writes (WRITE_TOOLS) only run after a person approves them in the graph's
approval step; the tool itself never decides that. A write that can't be done raises, so the
graph doesn't count it as done.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from order_agent import data

WRITE_TOOLS = {"cancel_order", "update_order_status"}
DATA_TOOLS = {"search_orders", "order_stats", "product_lookup", "research_clinic"}


@tool
def search_orders(clinic: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find orders, optionally filtered by clinic name (partial, case-blind) and status
    (processing, shipped, completed, cancelled)."""
    rows = data.ORDERS
    if clinic:
        rows = [o for o in rows if clinic.lower() in o["clinic"].lower()]
    if status:
        rows = [o for o in rows if o["status"] == status.lower()]
    return [data.describe(o) for o in rows]


@tool
def order_stats(clinic: Optional[str] = None) -> Dict[str, Any]:
    """Count orders and add up their value, by status, optionally for one clinic."""
    rows = [o for o in data.ORDERS if not clinic or clinic.lower() in o["clinic"].lower()]
    by_status: Dict[str, Dict[str, float]] = {}
    for o in rows:
        s = by_status.setdefault(o["status"], {"orders": 0, "value": 0.0})
        s["orders"] += 1
        s["value"] = round(s["value"] + data.order_total(o), 2)
    return {"orders": len(rows), "value": round(sum(data.order_total(o) for o in rows), 2),
            "by_status": by_status}


_TERM = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_FILLER = {"the", "a", "an", "of", "is", "in", "any", "do", "we", "have", "stock", "x", "mm"}


def _terms(text: str) -> set:
    """Words and sizes, in any order: "4.5 x 11.5 mm implant" -> {"4.5", "11.5", "implant"}."""
    return {t for t in _TERM.findall(text.lower()) if t not in _FILLER}


@tool
def product_lookup(query: str) -> List[Dict[str, Any]]:
    """Find catalog products by SKU, name or size, in any word order, with price and stock on hand.
    Returns the products that match the most terms of the query."""
    wanted = _terms(query)
    scored = [(len(wanted & _terms(f"{sku} {p['name']}")), sku) for sku, p in data.PRODUCTS.items()]
    best = max((n for n, _ in scored), default=0)
    return [{"sku": sku, **data.PRODUCTS[sku]} for n, sku in scored if best and n == best]


@tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel an order that has not shipped. Needs staff approval."""
    order = next((o for o in data.ORDERS if o["id"] == order_id.upper()), None)
    if order is None:
        raise ValueError(f"No order {order_id}.")
    if order["status"] in ("shipped", "completed"):
        raise ValueError(f"{order['id']} is already {order['status']}; it can't be cancelled.")
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    return f"{order['id']} cancelled."


@tool
def update_order_status(order_id: str, status: str) -> str:
    """Move an order to processing, shipped or completed. Needs staff approval."""
    if status not in ("processing", "shipped", "completed"):
        raise ValueError(f"Unknown status {status}.")
    order = next((o for o in data.ORDERS if o["id"] == order_id.upper()), None)
    if order is None:
        raise ValueError(f"No order {order_id}.")
    order["status"] = status
    return f"{order['id']} is now {status}."


READ_TOOLS = [search_orders, order_stats, product_lookup]
ACTION_TOOLS = [cancel_order, update_order_status]
