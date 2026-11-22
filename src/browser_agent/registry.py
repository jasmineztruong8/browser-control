"""Name-based registries for swappable components.

Configs refer to components by name (``env.name``, ``policy.name``, ``reward.type``, and the
``check`` field in task files). To add a new component, write it and decorate it::

    from browser_agent.registry import ENVS

    @ENVS.register("webarena_shop")
    class WebArenaShopEnv(PlaywrightEnv):
        ...

then add its module to ``PLUGIN_MODULES`` below so it is imported at startup.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")

# Modules that register components when imported. Add new modules here.
PLUGIN_MODULES = [
    "browser_agent.env.mock_shop",
    "browser_agent.policy",
    "browser_agent.rewards",
    "browser_agent.checks",
]


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        load_plugins()
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        load_plugins()
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        load_plugins()
        return name in self._items


ENVS: Registry[Any] = Registry("environment")
POLICIES: Registry[Any] = Registry("policy")
REWARDS: Registry[Any] = Registry("reward")
CHECKS: Registry[Any] = Registry("check")

_loaded = False


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in PLUGIN_MODULES:
        importlib.import_module(module)
