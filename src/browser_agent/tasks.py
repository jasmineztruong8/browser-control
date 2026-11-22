"""Task definitions and loading.

Tasks live in JSON files under ``data/tasks/``. Each task has an intent shown to the agent,
declarative ``success`` and ``subgoals`` checks evaluated on the environment state (see
``checks.py``), an optional ``initial_state`` the environment applies on reset, and an optional
scripted ``demo`` used to generate verified demonstrations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from browser_agent.registry import CHECKS

Split = Literal["train", "val", "test"]


class CheckSpec(BaseModel):
    """``{"check": "<registered name>", ...params}``. Subgoals also carry an ``id``."""

    model_config = ConfigDict(extra="allow")

    check: str
    id: str | None = None

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class DemoStep(BaseModel):
    """One scripted step. ``target`` is matched against element labels at run time."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["click", "type", "scroll", "stop"]
    target: str | None = None
    role: str | None = None  # optional role filter when several labels match
    text: str | None = None
    submit: bool = False
    direction: Literal["up", "down"] | None = None
    answer: str | None = None


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    intent: str
    family: str
    split: Split
    start_path: str | None = None  # appended to the env's start URL, e.g. "#cart"
    initial_state: dict[str, Any] = Field(default_factory=dict)
    success: list[CheckSpec]
    subgoals: list[CheckSpec] = Field(default_factory=list)
    demo: list[DemoStep] | None = None

    @model_validator(mode="after")
    def _validate(self) -> Task:
        if not self.success:
            raise ValueError(f"task {self.id}: needs at least one success check")
        ids = [s.id for s in self.subgoals]
        if any(i is None for i in ids):
            raise ValueError(f"task {self.id}: every subgoal needs an 'id'")
        if len(set(ids)) != len(ids):
            raise ValueError(f"task {self.id}: subgoal ids must be unique")
        for spec in [*self.success, *self.subgoals]:
            if spec.check not in CHECKS:
                raise ValueError(
                    f"task {self.id}: unknown check '{spec.check}'. "
                    f"Available: {', '.join(CHECKS.names())}"
                )
        return self


def load_tasks(
    path: str | Path,
    split: Split | None = None,
    families: list[str] | None = None,
    limit: int | None = None,
) -> list[Task]:
    raw = json.loads(Path(path).read_text())
    items = raw["tasks"] if isinstance(raw, dict) else raw
    tasks = [Task.model_validate(t) for t in items]

    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            raise ValueError(f"duplicate task id '{t.id}' in {path}")
        seen.add(t.id)

    if split:
        tasks = [t for t in tasks if t.split == split]
    if families:
        tasks = [t for t in tasks if t.family in families]
    if limit:
        tasks = tasks[:limit]
    return tasks


def run_check(spec: CheckSpec, state: dict[str, Any], answer: str | None = None) -> bool:
    return bool(CHECKS.get(spec.check)(state, answer=answer, **spec.params))
