import pytest

from browser_agent.config import ConfigError, load_config, parse_override
from browser_agent.registry import ENVS, POLICIES, REWARDS
from conftest import CONFIG_DIR


def test_base_defaults():
    cfg = load_config(CONFIG_DIR / "base.yaml", env_file=None)
    assert cfg.env.name == "mock_shop"
    assert cfg.reward.type == "binary"
    assert cfg.policy.model_name == "Qwen/Qwen2.5-3B-Instruct"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path, env_file=None)
    assert cfg.env.name in ENVS
    assert cfg.policy.name in POLICIES
    assert cfg.reward.type in REWARDS


def test_inherits_chain():
    cfg = load_config(CONFIG_DIR / "grpo_subgoal.yaml", env_file=None)
    assert cfg.run.name == "grpo_subgoal"
    assert cfg.training.method == "grpo"  # from grpo_binary.yaml
    assert cfg.policy.temperature == 0.7  # from rloo.yaml
    assert cfg.reward.type == "subgoal"
    assert cfg.env.max_steps == 12  # from base.yaml


def test_cli_overrides_are_typed():
    cfg = load_config(
        CONFIG_DIR / "base.yaml",
        [
            "env.max_steps=20",
            "reward.type=subgoal",
            "tasks.families=[add_to_cart]",
            "env.start_url=null",
        ],
        env_file=None,
    )
    assert cfg.env.max_steps == 20
    assert cfg.reward.type == "subgoal"
    assert cfg.tasks.families == ["add_to_cart"]
    assert cfg.env.start_url is None


def test_parse_override_nesting():
    assert parse_override("a.b.c=1") == {"a": {"b": {"c": 1}}}
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="env.max_step: unknown key"):
        load_config(CONFIG_DIR / "base.yaml", ["env.max_step=5"], env_file=None)


def test_unknown_component_rejected():
    with pytest.raises(ConfigError, match="reward.type: unknown reward 'shaped'"):
        load_config(CONFIG_DIR / "base.yaml", ["reward.type=shaped"], env_file=None)


def test_subgoal_cap_must_stay_below_success():
    with pytest.raises(ConfigError, match="subgoal_cap"):
        load_config(CONFIG_DIR / "base.yaml", ["reward.subgoal_cap=1.0"], env_file=None)


def test_env_var_expansion(monkeypatch, tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("env:\n  start_url: ${SHOP_URL}\nrun:\n  output_dir: ${OUT:-fallback}\n")
    monkeypatch.setenv("SHOP_URL", "http://localhost:7770")
    monkeypatch.delenv("OUT", raising=False)
    cfg = load_config(cfg_file, env_file=None)
    assert cfg.env.start_url == "http://localhost:7770"
    assert cfg.run.output_dir == "fallback"

    monkeypatch.delenv("SHOP_URL")
    with pytest.raises(ConfigError, match="SHOP_URL is not set"):
        load_config(cfg_file, env_file=None)


def test_env_file_is_loaded(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_RUNS", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MY_RUNS=/tmp/my-runs\n")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("run:\n  output_dir: ${MY_RUNS}\n")
    cfg = load_config(cfg_file, env_file=env_file)
    assert cfg.run.output_dir == "/tmp/my-runs"
