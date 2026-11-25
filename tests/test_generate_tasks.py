"""Tests for the generated mock-shop task set and the generator that produces it."""

import json

import pytest

from browser_agent import generate_tasks as gen
from browser_agent.policy import ScriptedPolicy
from browser_agent.rollout import make_env, run_episode
from browser_agent.tasks import load_tasks
from conftest import CONFIG_DIR, REPO_ROOT

V1_PATH = REPO_ROOT / "data" / "tasks" / "mock_shop_v1.json"
V1_RAW = json.loads(V1_PATH.read_text())
V1_TASKS = load_tasks(V1_PATH)
SPLITS = ("train", "val", "test")
FAMILIES = ("add_to_cart", "update_quantity", "remove_item")


def _target(task) -> tuple[str, str]:
    """The (product_id, option) a task is about, derived from its checks rather than its id."""
    expected = task.success[0].params["items"]
    initial = task.initial_state.get("cart", [])
    if task.family == "add_to_cart":
        (item,) = expected
    elif task.family == "remove_item":
        (item,) = [i for i in initial if i not in expected]
    else:
        (item,) = [i for i in initial if i not in expected]  # the one whose quantity changed
    return item["product_id"], item["option"]


# --------------------------------------------------------------------------- generator


def test_committed_file_matches_generator():
    meta = V1_RAW["generator"]
    regenerated = gen.generate_tasks(seed=meta["seed"], val_fraction=meta["val_fraction"])
    assert regenerated == V1_RAW["tasks"], (
        "data/tasks/mock_shop_v1.json is out of date; run python -m browser_agent.generate_tasks"
    )
    assert meta["held_out_variants"] == sorted(f"{p}:{o}" for p, o in gen.HELD_OUT)


def test_generator_is_deterministic_and_rejects_bad_fraction():
    assert gen.generate_tasks(seed=3) == gen.generate_tasks(seed=3)
    with pytest.raises(ValueError, match="val_fraction"):
        gen.generate_tasks(val_fraction=1.0)


def test_seed_only_moves_tasks_between_train_and_val():
    a = {t["id"]: t["split"] for t in gen.generate_tasks(seed=0)}
    b = {t["id"]: t["split"] for t in gen.generate_tasks(seed=1)}
    assert a.keys() == b.keys()
    assert {k for k, s in a.items() if s == "test"} == {k for k, s in b.items() if s == "test"}
    assert a != b  # some train/val assignments differ


def test_write_task_file_round_trips(tmp_path):
    out = tmp_path / "tasks.json"
    written = gen.write_task_file(out, seed=0, val_fraction=0.2)
    assert load_tasks(out)  # loads and validates
    assert json.loads(out.read_text())["tasks"] == written


# --------------------------------------------------------------------------- the v1 file


def test_every_split_covers_every_family():
    for split in SPLITS:
        families = {t.family for t in V1_TASKS if t.split == split}
        assert families == set(FAMILIES), f"split {split} is missing families"
    assert sum(t.split == "train" for t in V1_TASKS) >= 50  # roadmap: 50-100 demonstrations


def test_held_out_variants_only_appear_as_test_targets():
    for task in V1_TASKS:
        target = _target(task)
        if task.split == "test":
            assert target in gen.HELD_OUT, f"{task.id}: test task targets a non-held-out variant"
        else:
            assert target not in gen.HELD_OUT, f"{task.id}: held-out variant in {task.split}"


def test_expected_carts_follow_from_initial_state():
    """Success checks must be consistent with the initial cart, or demos could never pass."""
    for task in V1_TASKS:
        expected = task.success[0].params["items"]
        initial = task.initial_state.get("cart", [])
        if task.family == "add_to_cart":
            assert initial == [] and len(expected) == 1
        elif task.family == "remove_item":
            assert len(expected) == len(initial) - 1
            assert all(i in initial for i in expected)
        else:
            assert len(expected) == len(initial)
            changed = [i for i in initial if i not in expected]
            assert len(changed) == 1
        products = [i["product_id"] for i in initial]
        assert len(products) == len(set(products)), f"{task.id}: duplicate product in cart"


def test_ids_do_not_collide_with_sample_file():
    sample_ids = {t.id for t in load_tasks(REPO_ROOT / "data" / "tasks" / "mock_shop.json")}
    assert sample_ids.isdisjoint({t.id for t in V1_TASKS})


# --------------------------------------------------------------------------- browser


@pytest.fixture(scope="module")
def env_and_cfg():
    from browser_agent.config import load_config

    cfg = load_config(
        CONFIG_DIR / "base.yaml", [f"tasks.path={V1_PATH}", "env.settle_ms=0"], env_file=None
    )
    env = make_env(cfg)
    yield env, cfg
    env.close()


@pytest.mark.browser
def test_catalog_matches_mock_shop_page(env_and_cfg):
    env, _ = env_and_cfg
    env.reset(V1_TASKS[0])
    page_catalog = env.page.evaluate(
        "() => CATALOG.map(p => [p.id, p.name, p.optionName, p.options])"
    )
    ours = [[p.id, p.name, p.option_name, list(p.options)] for p in gen.CATALOG]
    assert page_catalog == ours


@pytest.mark.browser
@pytest.mark.parametrize("task", V1_TASKS, ids=lambda t: t.id)
def test_every_generated_demo_succeeds(env_and_cfg, task):
    env, cfg = env_and_cfg
    traj = run_episode(env, ScriptedPolicy(cfg), task, cfg)
    errors = [s.error for s in traj.steps if s.error]
    assert traj.success, f"errors={errors} final_state={traj.final_state}"
    assert not errors
    assert traj.terminated == "stop"
    assert set(traj.subgoals_hit) == {s.id for s in task.subgoals}
