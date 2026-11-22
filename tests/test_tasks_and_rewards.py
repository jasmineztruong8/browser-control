import json

import pytest

from browser_agent.config import RewardConfig
from browser_agent.registry import REWARDS
from browser_agent.rewards import SubgoalTracker, task_success
from browser_agent.tasks import load_tasks
from conftest import TASKS_PATH

CFG = RewardConfig()


@pytest.fixture
def mug_task():
    return next(t for t in load_tasks(TASKS_PATH) if t.id == "add_mug_blue_1")


def state(*items, page="home", search=""):
    return {
        "page": page,
        "last_search": search,
        "cart": [{"product_id": p, "option": o, "quantity": q} for p, o, q in items],
    }


def test_load_and_filter_tasks():
    all_tasks = load_tasks(TASKS_PATH)
    assert {t.split for t in all_tasks} == {"train", "val", "test"}
    assert all(t.split == "val" for t in load_tasks(TASKS_PATH, split="val"))
    families = load_tasks(TASKS_PATH, families=["remove_item"])
    assert families and all(t.family == "remove_item" for t in families)
    assert len(load_tasks(TASKS_PATH, limit=2)) == 2


def test_task_file_errors(tmp_path):
    base = json.loads(TASKS_PATH.read_text())["tasks"][0]
    bad = tmp_path / "bad.json"

    bad.write_text(json.dumps([base, base]))
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(bad)

    bad.write_text(json.dumps([{**base, "success": [{"check": "not_a_check"}]}]))
    with pytest.raises(ValueError, match="unknown check"):
        load_tasks(bad)


def test_success_requires_exact_cart(mug_task):
    assert task_success(mug_task, state(("mug", "Blue", 1)), None, True, CFG)
    assert not task_success(mug_task, state(("mug", "Red", 1)), None, True, CFG)  # wrong option
    assert not task_success(mug_task, state(("mug", "Blue", 2)), None, True, CFG)  # wrong qty
    assert not task_success(mug_task, state(("tshirt", "M", 1)), None, True, CFG)  # wrong product
    assert not task_success(
        mug_task, state(("mug", "Blue", 1), ("notebook", "1-pack", 1)), None, True, CFG
    )  # extra item


def test_false_completion_claim_fails(mug_task):
    # The agent says it's done, but the cart is empty.
    assert not task_success(mug_task, state(), "Added the blue mug", True, CFG)


def test_require_stop(mug_task):
    correct = state(("mug", "Blue", 1))
    assert not task_success(mug_task, correct, None, False, CFG)
    assert task_success(mug_task, correct, None, False, RewardConfig(require_stop=False))


def test_subgoals_count_once(mug_task):
    tracker = SubgoalTracker(mug_task)
    assert tracker.update(state(search="blue mug")) == ["searched"]
    assert tracker.update(state(search="blue mug")) == []  # repeating earns nothing
    assert tracker.update(state(page="product/mug", search="mug")) == ["on_product_page"]
    assert tracker.update(state(("mug", "Red", 1), page="product/mug")) == ["product_in_cart"]
    assert tracker.hit == ["searched", "on_product_page", "product_in_cart"]


def test_binary_reward(mug_task):
    fn = REWARDS.get("binary")
    assert fn(mug_task, state(("mug", "Blue", 1)), None, True, ["searched"], CFG).reward == 1.0
    assert fn(mug_task, state(), None, True, ["searched"], CFG).reward == 0.0


def test_subgoal_reward_is_bounded_and_success_dominates(mug_task):
    fn = REWARDS.get("subgoal")
    cfg = RewardConfig(subgoal_bonus=0.2, subgoal_cap=0.3)
    all_subgoals = ["searched", "on_product_page", "product_in_cart"]
    failed = fn(mug_task, state(("mug", "Red", 1)), None, True, all_subgoals, cfg)
    succeeded = fn(mug_task, state(("mug", "Blue", 1)), None, True, [], cfg)
    assert failed.reward == pytest.approx(0.3)  # 3 x 0.2 capped at 0.3
    assert not failed.success
    assert succeeded.reward > failed.reward
    duplicated = fn(mug_task, state(), None, True, ["searched", "searched"], cfg)
    assert duplicated.reward == pytest.approx(0.2)
