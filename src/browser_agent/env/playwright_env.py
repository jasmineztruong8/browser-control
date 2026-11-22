"""Generic Playwright-backed environment shared by the mock shop and the real site.

Subclasses override ``start_url``, ``prepare`` (apply a task's initial state) and
``get_state``. Each ``reset`` opens a fresh browser context, so cookies and storage never leak
between episodes.
"""

from __future__ import annotations

from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from browser_agent.config import Config
from browser_agent.env.base import BrowserEnv, StepResult
from browser_agent.observe import AGENT_ID_ATTR, observe
from browser_agent.tasks import Task
from browser_agent.types import Action, ActionKind, Observation


class PlaywrightEnv(BrowserEnv):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    # ---- hooks for subclasses -------------------------------------------------------------

    def start_url(self, task: Task) -> str:
        if not self.cfg.env.start_url:
            raise ValueError(f"env.start_url must be set for env '{self.cfg.env.name}'")
        return self.cfg.env.start_url + (task.start_path or "")

    def prepare(self, task: Task) -> None:
        """Apply ``task.initial_state`` after the start page loads. Default: nothing."""

    def get_state(self) -> dict[str, Any]:
        return {"url": self._page.url}

    # ---- BrowserEnv ---------------------------------------------------------------------

    @property
    def _page(self) -> Page:
        if self.page is None:
            raise RuntimeError("call reset() before using the environment")
        return self.page

    def _ensure_browser(self) -> Browser:
        if self._browser is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.cfg.env.headless)
        return self._browser

    def reset(self, task: Task) -> Observation:
        browser = self._ensure_browser()
        if self._context is not None:
            self._context.close()
        vp = self.cfg.env.viewport
        self._context = browser.new_context(viewport={"width": vp.width, "height": vp.height})
        self._context.set_default_timeout(self.cfg.env.action_timeout_ms)
        self._context.set_default_navigation_timeout(self.cfg.env.navigation_timeout_ms)
        self.page = self._context.new_page()
        self.page.goto(self.start_url(task))
        self.prepare(task)
        self._settle()
        return self.observe()

    def observe(self) -> Observation:
        return observe(self._page, self.cfg.observation)

    def step(self, action: Action) -> StepResult:
        if action.kind is ActionKind.STOP:
            return StepResult(observation=self.observe(), done=True)
        error = None
        try:
            self._execute(action)
            self._settle()
        except PlaywrightError as exc:
            error = str(exc).splitlines()[0][:200]
        return StepResult(observation=self.observe(), done=False, error=error)

    def _execute(self, action: Action) -> None:
        page = self._page
        if action.kind is ActionKind.SCROLL:
            dy = self.cfg.env.scroll_pixels * (1 if action.direction == "down" else -1)
            page.mouse.wheel(0, dy)
            return

        locator = page.locator(f'[{AGENT_ID_ATTR}="{action.element_id}"]')
        if locator.count() == 0:
            raise PlaywrightError(f"no element with id {action.element_id} on the current page")
        if action.kind is ActionKind.CLICK:
            locator.click()
        elif action.kind is ActionKind.TYPE:
            tag = locator.evaluate("el => el.tagName")
            if tag == "SELECT":
                locator.select_option(label=action.text or "")
            else:
                locator.fill(action.text or "")
            if action.submit:
                locator.press("Enter")

    def _settle(self) -> None:
        page = self._page
        try:
            page.wait_for_load_state("domcontentloaded")
            # Let queued events (e.g. hashchange re-renders) run before observing.
            page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )
        except PlaywrightError:
            pass
        if self.cfg.env.settle_ms:
            page.wait_for_timeout(self.cfg.env.settle_ms)

    def close(self) -> None:
        for closer in (self._context, self._browser):
            if closer is not None:
                try:
                    closer.close()
                except PlaywrightError:
                    pass
        if self._pw is not None:
            self._pw.stop()
        self._pw = self._browser = self._context = self.page = None
