"""Demonstrations -> SFT examples. No torch here, so it is tested in CI.

An SFT example is one step of a successful demo trajectory: the exact prompt messages the
policy saw and the text it produced. Because the prompt comes from the shared rollout, the
model is trained on the same prompts it gets at evaluation and RL time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from browser_agent.config import Config
from browser_agent.env.base import BrowserEnv
from browser_agent.rollout import make_policy, run_episode
from browser_agent.tasks import load_tasks
from browser_agent.types import Trajectory, load_jsonl


@dataclass(frozen=True)
class SFTExample:
    messages: list[dict[str, str]]
    target: str
    task_id: str
    step: int


def examples_from_trajectories(trajectories: list[Trajectory]) -> list[SFTExample]:
    """One example per valid step of each successful trajectory.

    Failed trajectories are skipped, and so are steps with an error (unparseable output or an
    action that failed in the browser), so the model never imitates a mistake.
    """
    examples = []
    for traj in trajectories:
        if not traj.success:
            continue
        for step in traj.steps:
            if step.action is None or step.error:
                continue
            if step.messages is None:
                raise ValueError(
                    f"trajectory {traj.task_id} has no saved prompts; "
                    "collect demos with run.save_prompts=true"
                )
            examples.append(
                SFTExample(
                    messages=step.messages,
                    target=step.model_output,
                    task_id=traj.task_id,
                    step=step.index,
                )
            )
    return examples


def collect_demos(env: BrowserEnv, cfg: Config, split: str, run_id: str = "") -> list[Trajectory]:
    """Replay each task's scripted demo on ``split`` in the live environment."""
    demo_cfg = cfg.model_copy(deep=True)
    demo_cfg.policy.name = "scripted"
    demo_cfg.tasks.split = split  # type: ignore[assignment]
    demo_cfg.run.save_prompts = True
    t = demo_cfg.tasks
    tasks = load_tasks(t.path, t.split, t.families, t.limit)
    policy = make_policy(demo_cfg)
    try:
        return [run_episode(env, policy, task, demo_cfg, run_id) for task in tasks]
    finally:
        policy.close()


def load_demos(path: str | Path) -> dict[str, list[Trajectory]]:
    """Demo trajectories from a ``trajectories.jsonl``, grouped by their task's split."""
    by_split: dict[str, list[Trajectory]] = {}
    for traj in load_jsonl(path):
        by_split.setdefault(traj.metadata.get("split", "train"), []).append(traj)
    return by_split
