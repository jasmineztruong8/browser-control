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


def _line_key(item: dict[str, Any]) -> tuple:
    return (item.get("product_id"), item.get("option"), item.get("quantity"))


@CHECKS.register("cart_contains")
def cart_contains(state, answer=None, *, product_id, option=None, quantity=None) -> bool:
    return any(_item_matches(i, product_id, option, quantity) for i in _cart(state))


@CHECKS.register("cart_not_contains")
def cart_not_contains(state, answer=None, *, product_id, option=None) -> bool:
    return not any(_item_matches(i, product_id, option, None) for i in _cart(state))


@CHECKS.register("cart_quantity")
def cart_quantity(state, answer=None, *, product_id, quantity, option=None) -> bool:
    """A cart line for ``product_id`` (and ``option``, if given) has exactly ``quantity``.

    Same as ``cart_contains`` with a required ``quantity``; named separately so "update
    quantity" subgoals read clearly in task JSON.
    """
    return cart_contains(state, answer, product_id=product_id, option=option, quantity=quantity)


@CHECKS.register("cart_others_unchanged")
def cart_others_unchanged(state, answer=None, *, items) -> bool:
    """Every line in ``items`` is still in the cart with the same product, option and quantity.

    ``items`` uses the same shape as ``cart_exact``: a list of
    ``{"product_id": ..., "option": ..., "quantity": ...}`` dicts, normally the task's
    ``initial_state.cart`` minus the line the task targets. Extra cart lines are ignored (pair
    with ``cart_exact`` for that). Matching is exact on all three fields (no wildcards). An
    empty ``items`` list is vacuously true.

    Holds at reset by construction, so use it as a ``success`` conjunct, not as a subgoal (a
    subgoal would pay its bonus for doing nothing).
    """
    cart = {_line_key(i) for i in _cart(state)}
    return all(_line_key(it) in cart for it in items)


@CHECKS.register("cart_exact")
def cart_exact(state, answer=None, *, items) -> bool:
    """Cart holds exactly ``items`` (order-insensitive): rejects wrong or extra items."""
    return sorted(map(_line_key, _cart(state))) == sorted(map(_line_key, items))


@CHECKS.register("page_is")
def page_is(state, answer=None, *, route) -> bool:
    return state.get("page") == route


@CHECKS.register("search_contains")
def search_contains(state, answer=None, *, text) -> bool:
    return text.lower() in (state.get("last_search") or "").lower()


@CHECKS.register("answer_contains")
def answer_contains(state, answer=None, *, text) -> bool:
    return answer is not None and text.lower() in answer.lower()
