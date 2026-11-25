"""Reward functions, selected by ``reward.type`` in the config.

All rewards share one signature so the RL trainers never need to know which one is active::

    fn(task, state, answer, stopped, subgoals_hit, cfg: RewardConfig) -> RewardResult

Task success is the same for every reward type, so conditions are always compared on it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from browser_agent.config import RewardConfig
from browser_agent.registry import REWARDS
from browser_agent.tasks import Task, run_check


class RewardResult(BaseModel):
    reward: float
    success: bool


def task_success(
    task: Task, state: dict[str, Any], answer: str | None, stopped: bool, cfg: RewardConfig
) -> bool:
    """All success checks pass on the final state (and the agent stopped, if required).

    Because checks read the real environment state, a "done" claim with the wrong product,
    option or quantity in the cart still fails.
    """
    if cfg.require_stop and not stopped:
        return False
    return all(run_check(spec, state, answer) for spec in task.success)


class SubgoalTracker:
    """Records the first step at which each subgoal holds. Each subgoal counts once."""

    def __init__(self, task: Task) -> None:
        self.task = task
        self.hit: list[str] = []

    def update(self, state: dict[str, Any], answer: str | None = None) -> list[str]:
        new = [
            s.id for s in self.task.subgoals if s.id not in self.hit and run_check(s, state, answer)
        ]
        self.hit.extend(new)  # type: ignore[arg-type]
        return new  # type: ignore[return-value]


@REWARDS.register("binary")
def binary_reward(task, state, answer, stopped, subgoals_hit, cfg: RewardConfig) -> RewardResult:
    success = task_success(task, state, answer, stopped, cfg)
    return RewardResult(reward=cfg.success_reward if success else 0.0, success=success)


@REWARDS.register("subgoal")
def subgoal_reward(task, state, answer, stopped, subgoals_hit, cfg: RewardConfig) -> RewardResult:
    """Binary success plus a bounded bonus per verified subgoal.

    The bonus is capped at ``subgoal_cap`` < ``success_reward`` (enforced by the config), so a
    failed episode can never out-score a successful one.

    With ``subgoal_normalize`` the bonus is ``subgoal_cap`` times the fraction of the task's own
    subgoals hit, so every task family can earn the same maximum bonus regardless of how many
    subgoals it defines (add-to-cart tasks have 3, update/remove tasks have 2).
    """
    success = task_success(task, state, answer, stopped, cfg)
    if cfg.subgoal_normalize:
        ids = {s.id for s in task.subgoals}
        frac = len(set(subgoals_hit) & ids) / len(ids) if ids else 0.0
        bonus = cfg.subgoal_cap * frac
    else:
        bonus = min(cfg.subgoal_cap, cfg.subgoal_bonus * len(set(subgoals_hit)))
    return RewardResult(reward=(cfg.success_reward if success else 0.0) + bonus, success=success)
