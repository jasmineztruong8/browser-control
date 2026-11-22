"""End-to-end tests against the local mock shop in headless Chromium."""

import pytest

from browser_agent.policy import Policy, PolicyOutput, ScriptedPolicy
from browser_agent.rollout import make_env, run_episode
from browser_agent.tasks import load_tasks
from browser_agent.types import Action, ActionKind
from conftest import CONFIG_DIR, TASKS_PATH

pytestmark = pytest.mark.browser
ALL_TASKS = load_tasks(TASKS_PATH)


@pytest.fixture(scope="module")
def env_and_cfg():
    from browser_agent.config import load_config

    cfg = load_config(
        CONFIG_DIR / "base.yaml", [f"tasks.path={TASKS_PATH}", "env.settle_ms=0"], env_file=None
    )
    env = make_env(cfg)
    yield env, cfg
    env.close()


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_scripted_demo_succeeds(env_and_cfg, task):
    env, cfg = env_and_cfg
    traj = run_episode(env, ScriptedPolicy(cfg), task, cfg)
    errors = [s.error for s in traj.steps if s.error]
    assert traj.success, f"failed with errors={errors} final_state={traj.final_state}"
    assert traj.reward == 1.0
    assert traj.terminated == "stop"
    assert not errors
    assert set(traj.subgoals_hit) == {s.id for s in task.subgoals}
    assert traj.steps[-1].action.kind is ActionKind.STOP
    assert traj.steps[0].messages[0]["role"] == "system"


class _Fixed(Policy):
    def __init__(self, cfg, outputs):
        super().__init__(cfg)
        self.outputs = outputs

    def act(self, messages, ctx):
        return PolicyOutput(text=self.outputs[min(ctx.step_index, len(self.outputs) - 1)])


def test_invalid_actions_end_episode(env_and_cfg):
    env, cfg = env_and_cfg
    traj = run_episode(env, _Fixed(cfg, ["I am not sure what to do"]), ALL_TASKS[0], cfg)
    assert traj.terminated == "invalid_limit"
    assert len(traj.steps) == cfg.agent.max_consecutive_invalid
    assert not traj.success


def test_bad_element_id_is_an_execution_error(env_and_cfg):
    env, cfg = env_and_cfg
    traj = run_episode(env, _Fixed(cfg, ["click [999]", "stop"]), ALL_TASKS[0], cfg)
    assert traj.steps[0].error.startswith("execution_error")
    assert traj.terminated == "stop"
    assert not traj.success


def test_premature_stop_fails(env_and_cfg):
    env, cfg = env_and_cfg
    traj = run_episode(env, _Fixed(cfg, ['stop "done"']), ALL_TASKS[0], cfg)
    assert traj.terminated == "stop"
    assert not traj.success


def test_form_error_appears_in_observation(env_and_cfg):
    env, cfg = env_and_cfg
    task = ALL_TASKS[0]
    obs = env.reset(task)
    env.page.goto(env.page.url.split("#")[0] + "#product/mug")
    obs = env.observe()
    add = next(e for e in obs.elements if e.label == "Add to Cart")

    result = env.step(Action(kind=ActionKind.CLICK, element_id=add.id))
    assert "Please choose a color." in result.observation.errors
    color = next(e for e in result.observation.elements if e.label == "Color")
    assert color.options == ["Blue", "Red", "Green"]
