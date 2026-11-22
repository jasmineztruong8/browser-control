from pathlib import Path

import pytest

from browser_agent.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
TASKS_PATH = REPO_ROOT / "data" / "tasks" / "mock_shop.json"


@pytest.fixture
def cfg():
    """Base config with repo-relative paths made absolute, no .env loading."""
    return load_config(
        CONFIG_DIR / "base.yaml",
        [f"tasks.path={TASKS_PATH}", "env.settle_ms=0"],
        env_file=None,
    )
