"""Browser environments. Register new ones with ``@ENVS.register("name")``."""

from browser_agent.env.base import BrowserEnv, StepResult

__all__ = ["BrowserEnv", "StepResult"]
