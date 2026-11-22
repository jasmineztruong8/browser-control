"""The episode loop shared by demonstration collection, RL training and evaluation.

Command line::

    python -m browser_agent.rollout --config experiments/configs/base.yaml \\
        --set tasks.split=val policy.name=scripted

Writes ``<run.output_dir>/<timestamp>-<run.name>/`` containing ``config.yaml`` (the fully
resolved config), ``trajectories.jsonl`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from browser_agent.actions import InvalidActionError, parse_action
from browser_agent.config import Config, ConfigError, load_config, save_config
from browser_agent.env.base import BrowserEnv
from browser_agent.policy import Policy, PolicyContext
from browser_agent.prompts import build_messages
from browser_agent.registry import ENVS, POLICIES, REWARDS
from browser_agent.rewards import SubgoalTracker
from browser_agent.tasks import Task, load_tasks
from browser_agent.types import ActionKind, StepRecord, Trajectory, save_jsonl

DEFAULT_CONFIG = "experiments/configs/base.yaml"


def make_env(cfg: Config) -> BrowserEnv:
    return ENVS.get(cfg.env.name)(cfg)


def make_policy(cfg: Config) -> Policy:
    return POLICIES.get(cfg.policy.name)(cfg)


def run_episode(
    env: BrowserEnv, policy: Policy, task: Task, cfg: Config, run_id: str = ""
) -> Trajectory:
    reward_fn = REWARDS.get(cfg.reward.type)
    tracker = SubgoalTracker(task)
    obs = env.reset(task)
    tracker.update(env.get_state())

    steps: list[StepRecord] = []
    answer: str | None = None
    stopped = False
    terminated = "max_steps"
    consecutive_errors = 0

    for i in range(cfg.env.max_steps):
        messages = build_messages(cfg, task, obs, steps)
        start = time.perf_counter()
        out = policy.act(
            messages, PolicyContext(task=task, observation=obs, step_index=i, history=steps)
        )
        step = StepRecord(
            index=i,
            observation=obs,
            messages=messages if cfg.run.save_prompts else None,
            model_output=out.text,
            latency_s=time.perf_counter() - start,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
        )
        try:
            action = parse_action(out.text, cfg.agent.allowed_actions)
        except InvalidActionError as exc:
            step.error = f"invalid_action: {exc}"
        else:
            step.action = action
            result = env.step(action)
            obs = result.observation
            if result.error:
                step.error = f"execution_error: {result.error}"
            if action.kind is ActionKind.STOP:
                stopped, answer = True, action.answer

        step.subgoals_hit = tracker.update(env.get_state(), answer)
        steps.append(step)
        consecutive_errors = consecutive_errors + 1 if step.error else 0

        if stopped:
            terminated = "stop"
            break
        if consecutive_errors >= cfg.agent.max_consecutive_invalid:
            terminated = "invalid_limit"
            break

    final_state = env.get_state()
    result = reward_fn(
        task=task,
        state=final_state,
        answer=answer,
        stopped=stopped,
        subgoals_hit=tracker.hit,
        cfg=cfg.reward,
    )
    return Trajectory(
        run_id=run_id,
        task_id=task.id,
        steps=steps,
        success=result.success,
        reward=result.reward,
        subgoals_hit=list(tracker.hit),
        terminated=terminated,  # type: ignore[arg-type]
        answer=answer,
        final_state=final_state,
        metadata={
            "family": task.family,
            "split": task.split,
            "env": cfg.env.name,
            "policy": cfg.policy.name,
            "model": cfg.policy.model_name if cfg.policy.name != "scripted" else None,
            "adapter": cfg.policy.adapter_path,
            "prompt_mode": cfg.prompt.mode,
            "reward_type": cfg.reward.type,
        },
    )


def summarize(trajectories: list[Trajectory]) -> dict[str, Any]:
    n = len(trajectories)
    total_steps = sum(len(t.steps) for t in trajectories)
    return {
        "episodes": n,
        "success_rate": sum(t.success for t in trajectories) / n if n else 0.0,
        "mean_reward": sum(t.reward for t in trajectories) / n if n else 0.0,
        "mean_steps": total_steps / n if n else 0.0,
        "invalid_action_rate": (
            sum(t.num_invalid for t in trajectories) / total_steps if total_steps else 0.0
        ),
        "mean_step_latency_s": (
            sum(s.latency_s for t in trajectories for s in t.steps) / total_steps
            if total_steps
            else 0.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run episodes and log trajectories.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    random.seed(cfg.run.seed)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.run.name}"
    run_dir = Path(cfg.run.output_dir) / run_id
    save_config(cfg, run_dir / "config.yaml")

    tasks = load_tasks(cfg.tasks.path, cfg.tasks.split, cfg.tasks.families, cfg.tasks.limit)
    if not tasks:
        print("No tasks matched the tasks.* filters.", file=sys.stderr)
        return 1

    trajectories: list[Trajectory] = []
    policy = make_policy(cfg)
    try:
        with make_env(cfg) as env:
            for task in tasks:
                traj = run_episode(env, policy, task, cfg, run_id)
                save_jsonl([traj], run_dir / "trajectories.jsonl", append=True)
                trajectories.append(traj)
                status = "OK  " if traj.success else "FAIL"
                print(
                    f"{status} {task.id:<28} reward={traj.reward:.2f} "
                    f"steps={len(traj.steps)} end={traj.terminated}"
                )
    finally:
        policy.close()

    summary = summarize(trajectories)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
