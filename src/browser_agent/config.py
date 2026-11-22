"""Typed, layered experiment configuration.

Configs are YAML files in ``experiments/configs/``. A file may start with ``inherits: base.yaml``
and then list only the keys it changes. Any key can be overridden on the command line with
``--set section.key=value``. String values may reference environment variables (loaded from
``.env``) as ``${NAME}`` or ``${NAME:-default}``. Every key is documented in
``docs/configuration.md``; unknown keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ActionName = Literal["click", "type", "scroll", "stop"]


class ConfigError(ValueError):
    pass


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Section):
    name: str = "default"
    output_dir: str = "runs"
    seed: int = 0
    save_prompts: bool = True


class ViewportConfig(_Section):
    width: int = 1280
    height: int = 800


class EnvConfig(_Section):
    name: str = "mock_shop"
    start_url: str | None = None
    headless: bool = True
    max_steps: int = Field(12, ge=1)
    action_timeout_ms: int = Field(5000, ge=1)
    navigation_timeout_ms: int = Field(15000, ge=1)
    settle_ms: int = Field(100, ge=0)
    scroll_pixels: int = Field(600, ge=1)
    viewport: ViewportConfig = Field(default_factory=ViewportConfig)


class ObservationConfig(_Section):
    max_elements: int = Field(60, ge=1)
    max_label_chars: int = Field(80, ge=1)
    max_options: int = Field(10, ge=0)
    include_page_text: bool = True
    max_page_text_chars: int = Field(600, ge=0)
    max_errors: int = Field(5, ge=0)


class AgentConfig(_Section):
    allowed_actions: list[ActionName] = Field(
        default_factory=lambda: ["click", "type", "scroll", "stop"]
    )
    history_length: int = Field(3, ge=0)
    max_consecutive_invalid: int = Field(3, ge=1)


class PromptConfig(_Section):
    mode: Literal["action_only", "plan_then_act"] = "action_only"
    template_dir: str | None = None  # None = built-in templates in src/browser_agent/prompts/


class PolicyConfig(_Section):
    name: str = "scripted"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path: str | None = None
    load_in_4bit: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = Field(96, ge=1)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)


class RewardConfig(_Section):
    type: str = "binary"
    success_reward: float = 1.0
    subgoal_bonus: float = 0.1
    subgoal_cap: float = 0.3
    require_stop: bool = True

    @model_validator(mode="after")
    def _success_dominates(self) -> RewardConfig:
        if self.subgoal_cap >= self.success_reward:
            raise ValueError(
                "reward.subgoal_cap must be smaller than reward.success_reward "
                "so that final success always dominates subgoal bonuses"
            )
        return self


class TasksConfig(_Section):
    path: str = "data/tasks/mock_shop.json"
    split: Literal["train", "val", "test"] | None = "train"
    families: list[str] | None = None
    limit: int | None = Field(None, ge=1)


class LoraConfig(_Section):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class TrainingConfig(_Section):
    method: Literal["none", "sft", "rloo", "grpo"] = "none"
    init_checkpoint: str | None = None
    output_dir: str = "checkpoints"
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 4
    group_size: int = 4  # rollouts per task for RLOO / GRPO
    max_env_interactions: int | None = None  # browser-step budget, matched across RL runs
    kl_coef: float = 0.0
    gradient_checkpointing: bool = True
    lora: LoraConfig = Field(default_factory=LoraConfig)


class Config(_Section):
    run: RunConfig = Field(default_factory=RunConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# --------------------------------------------------------------------------- loading

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name)
        if not resolved:
            if default is None:
                raise ConfigError(f"Environment variable {name} is not set and has no default")
            resolved = default
        return resolved

    expanded = _ENV_VAR.sub(replace, value)
    # A value that was only a variable and resolved to "" means "unset".
    if expanded == "" and _ENV_VAR.fullmatch(value):
        return None
    return expanded


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular 'inherits' chain: {' -> '.join(map(str, seen + (path,)))}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    parent = data.pop("inherits", None)
    if parent:
        data = deep_merge(_read_yaml(path.parent / parent, seen + (path,)), data)
    return data


def parse_override(item: str) -> dict[str, Any]:
    """Turn ``a.b.c=value`` into ``{"a": {"b": {"c": value}}}``; value is parsed as YAML."""
    if "=" not in item:
        raise ConfigError(f"Override '{item}' must look like section.key=value")
    dotted, raw = item.split("=", 1)
    keys = [k for k in dotted.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"Override '{item}' has an empty key")
    value: Any = yaml.safe_load(raw) if raw.strip() else None
    for key in reversed(keys):
        value = {key: value}
    return value


def _format_validation_error(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        msg = "unknown key" if e["type"] == "extra_forbidden" else e["msg"]
        lines.append(f"  {loc}: {msg}")
    return "Invalid config:\n" + "\n".join(lines)


def _check_component_names(cfg: Config) -> None:
    from browser_agent.registry import ENVS, POLICIES, REWARDS

    for registry, name, key in (
        (ENVS, cfg.env.name, "env.name"),
        (POLICIES, cfg.policy.name, "policy.name"),
        (REWARDS, cfg.reward.type, "reward.type"),
    ):
        if name not in registry:
            raise ConfigError(
                f"{key}: unknown {registry.kind} '{name}'. Available: {', '.join(registry.names())}"
            )


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
    env_file: str | Path | None = ".env",
) -> Config:
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)
    data: dict[str, Any] = _read_yaml(Path(path)) if path else {}
    for item in overrides:
        data = deep_merge(data, parse_override(item))
    data = _expand_env(data)
    try:
        cfg = Config.model_validate(data)
    except ValidationError as err:
        raise ConfigError(_format_validation_error(err)) from None
    _check_component_names(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
