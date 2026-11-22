"""Environment interface every browser environment implements.

``rollout.run_episode`` only talks to this interface, so the mock shop and the real WebArena
site are interchangeable via ``env.name`` in the config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from browser_agent.config import Config
from browser_agent.tasks import Task
from browser_agent.types import Action, Observation


@dataclass
class StepResult:
    observation: Observation
    done: bool
    error: str | None = None  # set when the action could not be executed
    info: dict[str, Any] = field(default_factory=dict)


class BrowserEnv(ABC):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def reset(self, task: Task) -> Observation:
        """Start a fresh episode for ``task`` (clean state, task's initial state applied)."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Execute one action. Must not raise for bad element ids or timeouts: report ``error``."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Site state used by success and subgoal checks (schema documented in checks.py)."""

    def close(self) -> None:
        pass

    def __enter__(self) -> BrowserEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
