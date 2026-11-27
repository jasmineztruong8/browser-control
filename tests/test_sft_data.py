"""SFT data pipeline and CLI checks. None of these need torch: the tokenizer test is skipped
unless the training extras are installed."""

import math

import pytest

from browser_agent.config import load_config
from browser_agent.rollout import make_env
from browser_agent.tasks import load_tasks
from browser_agent.training.data import (
    collect_demos,
    examples_from_trajectories,
    load_demos,
)
from browser_agent.training.sft import _selection_key, main
from browser_agent.types import Action, ActionKind, Observation, StepRecord, Trajectory, save_jsonl
from conftest import CONFIG_DIR, TASKS_PATH

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "page"}]


def _step(i, output="click [1]", action=True, error=None, messages=MESSAGES):
    return StepRecord(
        index=i,
        observation=Observation(url="file://x"),
        messages=messages,
        model_output=output,
        action=Action(kind=ActionKind.CLICK, element_id=1) if action else None,
        error=error,
    )


def test_examples_keep_only_valid_steps_of_successful_trajectories():
    good = Trajectory(
        task_id="a",
        success=True,
        steps=[
            _step(0),
            _step(1, output="garbage", action=False, error="invalid_action: x"),
            _step(2, error="execution_error: timeout"),
            _step(3, output="stop"),
        ],
    )
    failed = Trajectory(task_id="b", success=False, steps=[_step(0)])
    examples = examples_from_trajectories([good, failed])
    assert [(e.task_id, e.step, e.target) for e in examples] == [
        ("a", 0, "click [1]"),
        ("a", 3, "stop"),
    ]
    assert examples[0].messages == MESSAGES


def test_examples_require_saved_prompts():
    traj = Trajectory(task_id="a", success=True, steps=[_step(0, messages=None)])
    with pytest.raises(ValueError, match="save_prompts"):
        examples_from_trajectories([traj])


def test_load_demos_groups_by_split(tmp_path):
    trajs = [
        Trajectory(task_id="a", metadata={"split": "train"}),
        Trajectory(task_id="b", metadata={"split": "val"}),
        Trajectory(task_id="c", metadata={"split": "train"}),
    ]
    save_jsonl(trajs, tmp_path / "demos.jsonl")
    by_split = load_demos(tmp_path / "demos.jsonl")
    assert [t.task_id for t in by_split["train"]] == ["a", "c"]
    assert [t.task_id for t in by_split["val"]] == ["b"]


def test_selection_prefers_success_then_lower_loss():
    evals = [
        {"epoch": 1, "val_success_rate": 0.5, "val_loss": 0.1},
        {"epoch": 2, "val_success_rate": 0.75, "val_loss": 0.3},
        {"epoch": 3, "val_success_rate": 0.75, "val_loss": 0.2},
    ]
    assert max(evals, key=_selection_key)["epoch"] == 3
    loss_only = [{"epoch": 1, "val_loss": 0.4}, {"epoch": 2, "val_loss": 0.2}]
    assert max(loss_only, key=_selection_key)["epoch"] == 2
    assert _selection_key({}) == (-math.inf, -math.inf)


def test_sft_config_loads():
    cfg = load_config(CONFIG_DIR / "sft.yaml", env_file=None)
    t = cfg.training
    assert (t.method, cfg.policy.name, cfg.policy.temperature) == ("sft", "hf", 0.0)
    assert cfg.tasks.path.endswith("mock_shop_v1.json")
    assert t.demos_path is None and t.val_rollouts and 0 <= t.warmup_ratio < 1


@pytest.mark.parametrize(
    "overrides, message",
    [(["training.method=rloo"], "expected 'sft'"), (["policy.name=scripted"], "policy.name")],
)
def test_cli_rejects_wrong_method_or_policy(overrides, message, capsys):
    assert main(["--config", str(CONFIG_DIR / "sft.yaml"), "--set", *overrides]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.browser
def test_collect_demos_yields_examples_for_every_step(cfg):
    with make_env(cfg) as env:
        trajs = collect_demos(env, cfg, "val")
    tasks = load_tasks(TASKS_PATH, "val")
    assert [t.task_id for t in trajs] == [t.id for t in tasks]
    assert all(t.success and t.metadata["policy"] == "scripted" for t in trajs)
    examples = examples_from_trajectories(trajs)
    assert len(examples) == sum(len(t.demo) for t in tasks)
    assert examples[-1].target == "stop"
    assert all(e.messages[0]["role"] == "system" for e in examples)


def test_encode_masks_prompt_and_ends_the_turn():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from browser_agent.training.common import IGNORE_INDEX, collate, encode

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        )
    except OSError:
        pytest.skip("tokenizer not available offline")
    ids, labels = encode(tok, MESSAGES, 'type [2] "mug" enter')
    answer = [i for i, lab in zip(ids, labels, strict=True) if lab != IGNORE_INDEX]
    assert labels[: len(ids) - len(answer)] == [IGNORE_INDEX] * (len(ids) - len(answer))
    text = tok.decode(answer)
    assert text.startswith('type [2] "mug" enter') and "<|im_end|>" in text

    batch = collate([(ids, labels), ([1, 2], [IGNORE_INDEX, 2])], pad_id=0, device="cpu")
    assert batch["input_ids"].shape == (2, len(ids))
    assert batch["attention_mask"][1].sum() == 2
    assert (batch["labels"][1, 2:] == IGNORE_INDEX).all()
