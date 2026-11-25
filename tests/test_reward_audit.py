"""Reward audit: the new cart checks, reward invariants, and degenerate policies earning nothing."""

import pytest

from browser_agent.config import RewardConfig, load_config
from browser_agent.policy import Policy, PolicyOutput, find_element
from browser_agent.registry import CHECKS, REWARDS
from browser_agent.rewards import SubgoalTracker
from browser_agent.tasks import CheckSpec, load_tasks, run_check
from conftest import CONFIG_DIR, TASKS_PATH

ALL_TASKS = load_tasks(TASKS_PATH)


def state(*items, page="home", search=""):
    return {
        "page": page,
        "last_search": search,
        "cart": [{"product_id": p, "option": o, "quantity": q} for p, o, q in items],
    }


def line(p, o, q):
    return {"product_id": p, "option": o, "quantity": q}


# --- cart_quantity -------------------------------------------------------------------------


def test_cart_quantity():
    check = CHECKS.get("cart_quantity")
    s = state(("headphones", "Black", 3), ("mug", "Red", 1))
    assert check(s, product_id="headphones", quantity=3)
    assert check(s, product_id="headphones", option="Black", quantity=3)
    assert not check(s, product_id="headphones", option="White", quantity=3)  # wrong option
    assert not check(s, product_id="headphones", option="Black", quantity=1)  # wrong quantity
    assert not check(s, product_id="bottle", quantity=3)  # missing line
    assert not check(state(), product_id="headphones", quantity=3)  # empty cart


# --- cart_others_unchanged -----------------------------------------------------------------


def test_cart_others_unchanged():
    check = CHECKS.get("cart_others_unchanged")
    keep = [line("mug", "Red", 1), line("bottle", "1 L", 2)]
    assert check(state(("mug", "Red", 1), ("bottle", "1 L", 2)), items=keep)
    # Target line removed / changed, extra line added: others are still intact.
    assert check(state(("bottle", "1 L", 2), ("mug", "Red", 1), ("tshirt", "M", 4)), items=keep)
    assert not check(state(("bottle", "1 L", 2)), items=keep)  # other item removed
    assert not check(state(("mug", "Red", 2), ("bottle", "1 L", 2)), items=keep)  # qty changed
    assert not check(state(("mug", "Blue", 1), ("bottle", "1 L", 2)), items=keep)  # option changed
    assert not check(state(), items=keep)  # cart wiped
    assert check(state(), items=[])  # nothing to preserve


def test_new_checks_usable_from_task_json():
    """Check args come straight from task JSON, so validate via CheckSpec/run_check."""
    spec = CheckSpec.model_validate(
        {"check": "cart_others_unchanged", "items": [line("mug", "Red", 1)]}
    )
    assert run_check(spec, state(("mug", "Red", 1)))
    assert not run_check(spec, state())
    spec = CheckSpec.model_validate(
        {"check": "cart_quantity", "product_id": "headphones", "option": "Black", "quantity": 3}
    )
    assert run_check(spec, state(("headphones", "Black", 3)))
    assert not run_check(spec, state(("headphones", "Black", 1)))


# --- reward invariants ---------------------------------------------------------------------


@pytest.fixture
def mug_task():
    return next(t for t in ALL_TASKS if t.id == "add_mug_blue_1")


def test_repeated_subgoal_hits_do_not_grow_bonus(mug_task):
    cfg = RewardConfig()
    fn = REWARDS.get("subgoal")
    tracker = SubgoalTracker(mug_task)
    for _ in range(50):
        tracker.update(state(search="mug"))
    assert tracker.hit == ["searched"]
    once = fn(mug_task, state(), None, True, tracker.hit, cfg).reward
    assert once == pytest.approx(cfg.subgoal_bonus)
    # Even if a caller passes duplicates, set() de-duplicates them.
    spam = fn(mug_task, state(), None, True, ["searched"] * 50, cfg).reward
    assert spam == pytest.approx(once)


@pytest.mark.parametrize("bonus", [0.05, 0.1, 0.25, 0.9])
def test_bonus_never_exceeds_cap_and_failure_below_success(bonus):
    cfg = RewardConfig(subgoal_bonus=bonus, subgoal_cap=0.3)
    fn = REWARDS.get("subgoal")
    for task in ALL_TASKS:
        all_ids = [s.id for s in task.subgoals] + [f"extra_{i}" for i in range(20)]
        failed = fn(task, state(), None, True, all_ids, cfg)  # empty cart: never succeeds
        assert not failed.success
        assert failed.reward <= cfg.subgoal_cap + 1e-9
        assert failed.reward < cfg.success_reward


# --- browser-backed audit of degenerate policies -------------------------------------------


class _Repeat(Policy):
    """Emits the same text every step."""

    def __init__(self, cfg, text):
        super().__init__(cfg)
        self.text = text

    def act(self, messages, ctx):
        return PolicyOutput(text=self.text)


class _ClickCycle(Policy):
    """Clicks the nav links named in ``targets`` in turn, looked up by label each step.

    Ids are reassigned every observation, so the link is found by label rather than a fixed id.
    """

    def __init__(self, cfg, targets):
        super().__init__(cfg)
        self.targets = targets

    def act(self, messages, ctx):
        target = self.targets[ctx.step_index % len(self.targets)]
        el = find_element(ctx.observation.elements, target, role="link")
        assert el is not None, f"no {target!r} link on page"
        return PolicyOutput(text=f"click [{el.id}]")


AUDIT_MAX_STEPS = 5


@pytest.fixture(scope="module")
def audit_env_and_cfg():
    pytest.importorskip("playwright.sync_api")
    from browser_agent.rollout import make_env

    cfg = load_config(
        CONFIG_DIR / "base.yaml",
        [
            f"tasks.path={TASKS_PATH}",
            "tasks.split=null",
            "env.settle_ms=0",
            f"env.max_steps={AUDIT_MAX_STEPS}",
            # Let junk run to max_steps instead of stopping at the invalid limit.
            f"agent.max_consecutive_invalid={AUDIT_MAX_STEPS + 1}",
            "reward.type=subgoal",
            "run.save_prompts=false",
        ],
        env_file=None,
    )
    env = make_env(cfg)
    try:
        env.reset(ALL_TASKS[0])
    except Exception as exc:  # Chromium not installed, sandbox refuses to launch, etc.
        env.close()
        pytest.skip(f"Playwright Chromium unavailable: {exc}")
    yield env, cfg
    env.close()


@pytest.mark.browser
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_no_subgoal_satisfied_at_reset(audit_env_and_cfg, task):
    env, _ = audit_env_and_cfg
    env.reset(task)
    s = env.get_state()
    already = [sg.id for sg in task.subgoals if run_check(sg, s)]
    assert not already, f"subgoals true at reset: {already}"


@pytest.mark.browser
@pytest.mark.parametrize(
    "text,expected_end",
    [("stop", "stop"), ("I am not sure what to do", "max_steps"), ("click [0]", "max_steps")],
    ids=["stop-immediately", "spam-invalid", "spam-click-0"],
)
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_degenerate_policies_earn_nothing(audit_env_and_cfg, task, text, expected_end):
    from browser_agent.rollout import run_episode

    env, cfg = audit_env_and_cfg
    traj = run_episode(env, _Repeat(cfg, text), task, cfg)
    assert traj.terminated == expected_end
    assert not traj.success
    assert traj.subgoals_hit == []
    assert traj.reward == 0.0


@pytest.mark.browser
@pytest.mark.parametrize(
    "targets",
    [["Cart"], ["Cart", "Home"]],
    ids=["spam-cart-link", "cart-home-cart"],
)
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_cart_link_spam_earns_at_most_one_bonus(audit_env_and_cfg, task, targets):
    """Visiting the cart is one click away; repeating it (or leaving and returning) pays once."""
    from browser_agent.rollout import run_episode

    env, cfg = audit_env_and_cfg
    traj = run_episode(env, _ClickCycle(cfg, targets), task, cfg)
    assert traj.terminated == "max_steps"
    assert not traj.success
    has_cart_subgoal = any(sg.id == "on_cart_page" for sg in task.subgoals)
    assert traj.subgoals_hit == (["on_cart_page"] if has_cart_subgoal else [])
    expected = cfg.reward.subgoal_bonus if has_cart_subgoal else 0.0
    assert traj.reward == pytest.approx(expected)


# --- subgoal design: option-aware progress and equal shaping across families ---------------


def test_wrong_variant_earns_no_product_in_cart_bonus(mug_task):
    tracker = SubgoalTracker(mug_task)
    tracker.update(state(("mug", "Red", 1)))  # task asks for Blue
    assert "product_in_cart" not in tracker.hit
    tracker.update(state(("mug", "Blue", 1)))
    assert "product_in_cart" in tracker.hit


def test_every_task_has_at_least_two_subgoals():
    for task in ALL_TASKS:
        assert len(task.subgoals) >= 2, task.id


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_normalized_bonus_is_equal_across_families(task):
    cfg = RewardConfig(subgoal_normalize=True)
    fn = REWARDS.get("subgoal")
    ids = [s.id for s in task.subgoals]
    full = fn(task, state(), None, True, ids, cfg)  # empty cart: no success
    assert not full.success
    assert full.reward == pytest.approx(cfg.subgoal_cap)
    one = fn(task, state(), None, True, ids[:1] * 10, cfg).reward  # repeats count once
    assert one == pytest.approx(cfg.subgoal_cap / len(ids))
    # Ids that are not the task's own subgoals earn nothing.
    assert fn(task, state(), None, True, ["bogus", "extra"], cfg).reward == 0.0
