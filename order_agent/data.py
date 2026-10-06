"""A small in-memory order book. Made-up clinics and products; stands in for a real ERP."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

PRODUCTS: Dict[str, Dict[str, Any]] = {
    "IMP-4010": {"name": "Implant 4.0 x 10 mm", "price": 210.0, "stock": 140},
    "IMP-4511": {"name": "Implant 4.5 x 11.5 mm", "price": 225.0, "stock": 0},
    "ABT-STR": {"name": "Straight abutment", "price": 95.0, "stock": 320},
    "ABT-ANG": {"name": "Angled abutment 15 deg", "price": 120.0, "stock": 45},
    "HEA-45": {"name": "Healing cap 4.5", "price": 18.0, "stock": 800},
}

_SEED_ORDERS: List[Dict[str, Any]] = [
    {"id": "SO-1001", "clinic": "Bayview Dental", "status": "processing",
     "lines": [("IMP-4010", 4), ("ABT-STR", 4)]},
    {"id": "SO-1002", "clinic": "Bayview Dental", "status": "shipped",
     "lines": [("HEA-45", 20)], "tracking": "1Z999AA10123456784"},
    {"id": "SO-1003", "clinic": "Elm Street Smiles", "status": "processing",
     "lines": [("IMP-4511", 2), ("ABT-ANG", 2)]},
    {"id": "SO-1004", "clinic": "Elm Street Smiles", "status": "completed",
     "lines": [("IMP-4010", 10)]},
    {"id": "SO-1005", "clinic": "Harbor Oral Surgery", "status": "processing",
     "lines": [("ABT-ANG", 6), ("HEA-45", 6)]},
]

ORDERS: List[Dict[str, Any]] = []
DONE_ACTIONS: set = set()  # each approved write runs once, keyed by tool-call id


def reset() -> None:
    ORDERS[:] = copy.deepcopy(_SEED_ORDERS)
    DONE_ACTIONS.clear()


def order_total(order: Dict[str, Any]) -> float:
    return sum(PRODUCTS[sku]["price"] * qty for sku, qty in order["lines"])


def describe(order: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": order["id"],
        "clinic": order["clinic"],
        "status": order["status"],
        "lines": [{"sku": s, "name": PRODUCTS[s]["name"], "qty": q} for s, q in order["lines"]],
        "total": round(order_total(order), 2),
        **({"tracking": order["tracking"]} if order.get("tracking") else {}),
    }


reset()
