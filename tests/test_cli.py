"""CLI tests. Kept in their own module: Playwright allows one sync browser per thread, and
test_rollout_smoke.py holds one open for the whole module."""

import json

import pytest

from browser_agent.rollout import main, summarize
from browser_agent.tasks import load_tasks
from browser_agent.types import load_jsonl
from conftest import CONFIG_DIR, TASKS_PATH

ALL_TASKS = load_tasks(TASKS_PATH)


@pytest.mark.browser
def test_cli_writes_run_directory(tmp_path, capsys):
    code = main(
        [
            "--config",
            str(CONFIG_DIR / "base.yaml"),
            "--set",
            f"tasks.path={TASKS_PATH}",
            f"run.output_dir={tmp_path}",
            "tasks.split=val",
        ]
    )
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    assert (run_dir / "config.yaml").exists()
    trajs = load_jsonl(run_dir / "trajectories.jsonl")
    assert len(trajs) == len([t for t in ALL_TASKS if t.split == "val"])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary == pytest.approx(summarize(trajs))
    assert summary["success_rate"] == 1.0


def test_cli_reports_config_errors(capsys):
    assert main(["--config", str(CONFIG_DIR / "base.yaml"), "--set", "env.max_step=3"]) == 2
    assert "unknown key" in capsys.readouterr().err
