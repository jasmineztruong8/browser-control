"""Shared data types: the contract between environment, policy, rewards, training and eval.

A ``Trajectory`` is the unit everything exchanges: rollouts produce it, SFT reads it as
demonstrations, RLOO/GRPO read its rewards, and the evaluator summarizes it. Trajectories are
stored one per line in JSONL files (see ``save_jsonl`` / ``load_jsonl``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionKind(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    STOP = "stop"


class Action(BaseModel):
    """A structured browser action. ``to_text`` gives the canonical string the model emits."""

    kind: ActionKind
    element_id: int | None = None
    text: str | None = None
    submit: bool = False  # for TYPE: press Enter after typing
    direction: Literal["up", "down"] | None = None
    answer: str | None = None  # for STOP: optional final answer / completion claim

    def to_text(self) -> str:
        if self.kind is ActionKind.CLICK:
            return f"click [{self.element_id}]"
        if self.kind is ActionKind.TYPE:
            text = (self.text or "").replace('"', '\\"')
            return f'type [{self.element_id}] "{text}"' + (" enter" if self.submit else "")
        if self.kind is ActionKind.SCROLL:
            return f"scroll {self.direction}"
        if self.answer:
            return 'stop "{}"'.format(self.answer.replace('"', '\\"'))
        return "stop"


class Element(BaseModel):
    """One actionable element on the page. ``id`` is only stable within one observation."""

    id: int
    role: str
    label: str
    value: str | None = None
    options: list[str] | None = None  # for select / combobox elements


class Observation(BaseModel):
    url: str
    title: str = ""
    elements: list[Element] = Field(default_factory=list)
    page_text: str = ""
    errors: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    index: int
    observation: Observation
    messages: list[dict[str, str]] | None = None  # exact prompt given to the policy
    model_output: str
    action: Action | None = None  # None when the output could not be parsed
    error: str | None = None  # invalid action or execution failure
    subgoals_hit: list[str] = Field(default_factory=list)  # subgoals first reached at this step
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


TerminationReason = Literal["stop", "max_steps", "invalid_limit"]


class Trajectory(BaseModel):
    run_id: str = ""
    task_id: str
    steps: list[StepRecord] = Field(default_factory=list)
    success: bool = False
    reward: float = 0.0
    subgoals_hit: list[str] = Field(default_factory=list)
    terminated: TerminationReason = "max_steps"
    answer: str | None = None
    final_state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_invalid(self) -> int:
        return sum(1 for s in self.steps if s.error)


def save_jsonl(trajectories: Iterable[Trajectory], path: str | Path, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(traj.model_dump_json() + "\n")


def load_jsonl(path: str | Path) -> list[Trajectory]:
    with Path(path).open(encoding="utf-8") as f:
        return [Trajectory.model_validate(json.loads(line)) for line in f if line.strip()]
