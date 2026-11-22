# browser-control

COMS6998 Intro to DL and LLM Based GenAI Systems Final Semester Project

A browser-control agent for a self-hosted WebArena shopping site. We study how policy optimization (RLOO vs. GRPO), reward design (binary vs. bounded subgoal rewards) and reasoning prompts (action-only vs. plan-then-act) affect task completion, starting from a supervised fine-tuning (SFT) baseline on Qwen2.5-3B-Instruct.

Team: Jasmine Truong, Gia Yang, Nicolas Hunt

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core package + tests
playwright install chromium        # headless browser
cp .env.example .env               # then edit values for your machine
```

On the GPU machine, also install the training dependencies:

```bash
pip install -e ".[dev,train]"
```

## Quickstart

Run the scripted policy on the local mock shop. This needs no GPU and no WebArena:

```bash
python -m browser_agent.rollout                              # train split, defaults from base.yaml
python -m browser_agent.rollout --set tasks.split=val        # override any config key
python -m browser_agent.rollout --set env.headless=false env.settle_ms=800   # watch it in a browser
```

Each run writes `runs/<run_id>/` with the resolved `config.yaml`, `trajectories.jsonl` and `summary.json`.

Run the tests:

```bash
pytest -q
ruff check src tests
```

## Repository layout

```
src/browser_agent/      shared code used by experiments and the demo
  types.py              Action / Observation / Trajectory data types + JSONL I/O
  config.py             layered YAML configs with CLI and env-var overrides
  registry.py           name-based registries for swappable components
  env/                  BrowserEnv interface, Playwright implementation, mock shop
  observe.py            DOM -> compact observation
  actions.py            model text -> structured action
  prompts/              editable prompt templates (action_only, plan_then_act)
  policy.py             scripted and Hugging Face policies
  tasks.py, checks.py   task schema and success/subgoal checks
  rewards.py            binary and subgoal rewards
  rollout.py            episode loop + CLI
experiments/configs/    base.yaml + one config per experimental condition
data/                   task files and data documentation
demo/                   runnable demo
results/                curated metrics and figures for the report
tests/                  unit, integration and browser smoke tests
docs/                   project documentation
```

## Docs

- [Project proposal](docs/proposal.md): the approved final project proposal
- [Project deliverables](docs/project-deliverables.md): course specs with the grading breakdown, deadlines and rubrics
- [Roadmap](docs/roadmap.md): schedule, team responsibilities and experiment plan
- [Architecture](docs/architecture.md): how the components fit together and who owns each
- [Configuration](docs/configuration.md): every config key, its default and how to override it
- [Data](data/README.md): task file format and how to generate demonstrations
