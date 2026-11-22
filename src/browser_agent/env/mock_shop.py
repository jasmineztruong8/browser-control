"""A tiny local shopping site for development and tests, no WebArena or network needed.

The site is a single HTML file (``fixtures/mock_shop.html``) with search, product pages with
an option dropdown and quantity field, form validation errors, and a cart where quantities can
be updated or items removed. Set ``env.start_url`` to point at a different copy of the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from browser_agent.env.playwright_env import PlaywrightEnv
from browser_agent.registry import ENVS
from browser_agent.tasks import Task

DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "mock_shop.html"


@ENVS.register("mock_shop")
class MockShopEnv(PlaywrightEnv):
    def start_url(self, task: Task) -> str:
        base = self.cfg.env.start_url or DEFAULT_FIXTURE.resolve().as_uri()
        return base + (task.start_path or "")

    def prepare(self, task: Task) -> None:
        if task.initial_state:
            self._page.evaluate("s => window.shopApi.setState(s)", task.initial_state)

    def get_state(self) -> dict[str, Any]:
        state = self._page.evaluate("() => window.shopApi.getState()")
        state["url"] = self._page.url
        return state
