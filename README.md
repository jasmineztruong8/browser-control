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

## SFT baseline (Condition A)

On the GPU machine (e.g. the GCP VM, see [docs/gcp.md](docs/gcp.md)):

```bash
python -m browser_agent.training.sft --config experiments/configs/sft.yaml
```

This replays the scripted demos on the `train` and `val` tasks of `mock_shop_v1.json`, and trains a LoRA adapter on every demo step, with the loss on the action tokens only. After each epoch it runs the `val` tasks greedily in the browser. It keeps the epoch with the highest validation success and copies it to `checkpoints/sft`, the checkpoint every RL config starts from. Epoch 0 in the logs is the untrained base model, for reference. With the default settings (3 epochs, about 300 examples) it takes a few minutes on an L4. Logs go to `runs/<run_id>/`: `metrics.jsonl`, `val_epoch<k>.jsonl` and `summary.json`.

Evaluate the adapter on held-out tasks with the normal rollout command:

```bash
python -m browser_agent.rollout --config experiments/configs/sft.yaml --set policy.adapter_path=checkpoints/sft tasks.split=test
```

## Compute pilot

[`notebooks/compute_pilot.ipynb`](notebooks/compute_pilot.ipynb) checks that the model trains on a Colab GPU and measures the cost of each stage: memory, SFT step time, rollout and RL-update time, and an adapter save/reload. It ends with RL interactions per GPU hour and the GPU hours for a given budget. Results are written to `runs/<timestamp>-pilot/`. You can run it in two places:

- **Google Cloud:** `scripts/gcp/vm.sh create`, then `setup`, then `pilot`. See [docs/gcp.md](docs/gcp.md) for the one-time project and GPU quota setup. The same VM is used for training.
- **Colab:** *File → Open notebook → GitHub*, paste this repo's URL, select a GPU runtime and choose *Run all*.

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
  generate_tasks.py     mock-shop task generator with versioned train/val/test splits
  rewards.py            binary and subgoal rewards
  rollout.py            episode loop + CLI
  training/             SFT trainer (RLOO / GRPO next) and shared model/tokenization helpers
experiments/configs/    base.yaml + one config per experimental condition
data/                   task files and data documentation
demo/                   runnable demo
results/                curated metrics and figures for the report
notebooks/              notebooks (compute pilot), runnable on Colab or a GCP VM
scripts/gcp/            create and manage the team's GPU VM on Google Cloud
tests/                  unit, integration and browser smoke tests
docs/                   project documentation
```

## Docs

- [Project proposal](docs/proposal.md): the approved final project proposal
- [Project deliverables](docs/project-deliverables.md): course specs with the grading breakdown, deadlines and rubrics
- [Roadmap](docs/roadmap.md): schedule, team responsibilities and experiment plan
- [Architecture](docs/architecture.md): how the components fit together and who owns each
- [Configuration](docs/configuration.md): every config key, its default and how to override it
- [Google Cloud](docs/gcp.md): setting up the project and running the GPU VM
- [Data](data/README.md): task file format and how to generate demonstrations
