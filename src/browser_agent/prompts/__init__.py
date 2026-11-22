"""Prompt construction from editable templates.

Templates are plain text files using ``$name`` placeholders (Python ``string.Template``):

- ``system_<mode>.txt`` for each ``prompt.mode`` (``action_only``, ``plan_then_act``); gets ``$actions``
- ``user.txt``; gets ``$intent``, ``$history`` and ``$observation``

Set ``prompt.template_dir`` in a config to use a copy of these files with different wording.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from browser_agent.config import Config
from browser_agent.observe import format_observation
from browser_agent.tasks import Task
from browser_agent.types import Observation, StepRecord

DEFAULT_TEMPLATE_DIR = Path(__file__).parent

ACTION_HELP = {
    "click": "click [id]                 click an element",
    "type": 'type [id] "text"           type into a field or pick a dropdown option '
    "(append `enter` to press Enter)",
    "scroll": "scroll up | scroll down    scroll the page",
    "stop": 'stop  or  stop "answer"    finish the task',
}


def _template(cfg: Config, name: str) -> Template:
    directory = Path(cfg.prompt.template_dir) if cfg.prompt.template_dir else DEFAULT_TEMPLATE_DIR
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return Template(path.read_text())


def format_history(steps: list[StepRecord], limit: int) -> str:
    if limit == 0 or not steps:
        return "(none)"
    lines = []
    for s in steps[-limit:]:
        action = s.action.to_text() if s.action else s.model_output.strip()[:80]
        line = f"{s.index + 1}. {action}"
        if s.error:
            line += f"  -> error: {s.error}"
        lines.append(line)
    return "\n".join(lines)


def build_messages(
    cfg: Config, task: Task, observation: Observation, history: list[StepRecord]
) -> list[dict[str, str]]:
    actions = "\n".join(ACTION_HELP[a] for a in cfg.agent.allowed_actions)
    system = _template(cfg, f"system_{cfg.prompt.mode}.txt").substitute(actions=actions)
    user = _template(cfg, "user.txt").substitute(
        intent=task.intent,
        history=format_history(history, cfg.agent.history_length),
        observation=format_observation(observation),
    )
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
