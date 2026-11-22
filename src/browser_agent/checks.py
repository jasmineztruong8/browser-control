"""Declarative checks used by task ``success`` and ``subgoals`` lists.

Each check takes the environment state returned by ``BrowserEnv.get_state()`` plus the agent's
final ``answer`` (if it stopped with one) and returns True/False. The state schema these checks
expect is::

    {"page": "product/mug", "cart": [{"product_id": "mug", "option": "Blue", "quantity": 1}],
     "last_search": "mug", "url": "..."}

Environments should return state in this shape so tasks and rewards work unchanged. To add a
check, write a function and decorate it with ``@CHECKS.register("name")``.
"""

from __future__ import annotations

from typing import Any

from browser_agent.registry import CHECKS


def _cart(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(state.get("cart") or [])


def _item_matches(item: dict[str, Any], product_id: str, option: str | None, quantity: int | None):
    return (
        item.get("product_id") == product_id
        and (option is None or item.get("option") == option)
        and (quantity is None or item.get("quantity") == quantity)
    )


@CHECKS.register("cart_contains")
def cart_contains(state, answer=None, *, product_id, option=None, quantity=None) -> bool:
    return any(_item_matches(i, product_id, option, quantity) for i in _cart(state))


@CHECKS.register("cart_not_contains")
def cart_not_contains(state, answer=None, *, product_id, option=None) -> bool:
    return not any(_item_matches(i, product_id, option, None) for i in _cart(state))


@CHECKS.register("cart_exact")
def cart_exact(state, answer=None, *, items) -> bool:
    """Cart holds exactly ``items`` (order-insensitive): rejects wrong or extra items."""

    def key(i: dict[str, Any]) -> tuple:
        return (i.get("product_id"), i.get("option"), i.get("quantity"))

    return sorted(map(key, _cart(state))) == sorted(map(key, items))


@CHECKS.register("page_is")
def page_is(state, answer=None, *, route) -> bool:
    return state.get("page") == route


@CHECKS.register("search_contains")
def search_contains(state, answer=None, *, text) -> bool:
    return text.lower() in (state.get("last_search") or "").lower()


@CHECKS.register("answer_contains")
def answer_contains(state, answer=None, *, text) -> bool:
    return answer is not None and text.lower() in answer.lower()
