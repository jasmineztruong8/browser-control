# Architecture

Every part of the project (collecting demonstrations, SFT, RLOO, GRPO, evaluation and the demo) runs the same episode loop, `run_episode` in `src/browser_agent/rollout.py`. Each part plugs a different policy or config into that loop.

```
            ┌──────────── Task (data/tasks/*.json) ────────────┐
            │ intent, initial_state, success checks, subgoals  │
            └───────────────────────┬──────────────────────────┘
                                    │ reset(task)
                                    ▼
┌────────────┐  Observation   ┌───────────┐  messages   ┌──────────┐
│ BrowserEnv │ ─────────────▶ │  prompts  │ ──────────▶ │  Policy  │
│ (env/)     │                │ (prompts/)│             │(policy.py)│
└────────────┘                └───────────┘             └────┬─────┘
      ▲                                                       │ raw text
      │ step(action)          ┌────────────┐                  ▼
      └────────────────────── │ actions.py │ ◀──────── parse_action
                              └────────────┘
      │ get_state()
      ▼
┌──────────────┐  each step: SubgoalTracker   end: reward fn  ┌────────────┐
│ checks.py    │ ───────────────────────────────────────────▶ │ Trajectory │ ─▶ runs/<id>/trajectories.jsonl
│ rewards.py   │                                              │ (types.py) │
└──────────────┘                                              └────────────┘
```

## Components

| Module | Role |
| --- | --- |
| `types.py` | Shared data types: `Action`, `Observation`, `StepRecord`, `Trajectory`, JSONL I/O |
| `config.py` | Typed, layered YAML configs with CLI and environment-variable overrides |
| `registry.py` | Name-based registries so configs can swap environments, policies, rewards and checks |
| `env/base.py` | `BrowserEnv` interface: `reset`, `step`, `get_state` |
| `env/playwright_env.py` | Playwright implementation shared by all sites: element ids, action execution, timeouts |
| `env/mock_shop.py` | Local mock shopping site for development and tests |
| `observe.py` | DOM → compact `Observation` (actionable elements, form errors, page text) and its text rendering |
| `tasks.py`, `checks.py` | Task schema and loading; declarative success and subgoal checks on site state |
| `rewards.py` | `binary` and `subgoal` rewards and the subgoal tracker |
| `prompts/` | Editable prompt templates for `action_only` and `plan_then_act` |
| `policy.py` | `Policy` interface, `scripted` demo replayer, `hf` Hugging Face model |
| `rollout.py` | Episode loop, run summary and command-line entry point |
| `training/data.py` | Demo trajectories → SFT examples (prompt messages → action text). No torch |
| `training/common.py` | Trainable LoRA policy, prompt-masked tokenization, batching, LR schedule. Shared by the SFT and RL trainers |
| `training/sft.py` | Condition A trainer and CLI: collect demos, train, select the epoch by validation success |

## Key design decisions

- **One trajectory format for everything.** Each step stores the observation, the exact prompt messages, the raw model output, the parsed action and any error. SFT trains on (messages → output) pairs from successful demo trajectories, and RLOO and GRPO use the same records plus the episode reward. That means RL implementations can't drift apart in how they build prompts.
- **Rewards read site state, not model claims.** Success checks run on `env.get_state()`. So a `stop "done"` with the wrong product, option or quantity in the cart still fails, and `cart_exact` also rejects extra items.
- **Subgoal rewards are bounded.** Each subgoal counts once, and the total bonus is capped below the success reward. The config refuses to load otherwise, so a failed episode can never outscore a successful one.
- **Invalid actions don't crash an episode.** Unparseable output and actions that fail in the browser (for example a missing element id or a timeout) are recorded as step errors. The episode ends after `agent.max_consecutive_invalid` errors in a row.
- **Element ids are per observation.** Ids are reassigned in DOM order every time the page is observed, so the model only ever refers to what it was just shown.
- **The mock shop stands in for WebArena during development.** It has the same state schema the checks expect, so tasks, rewards and the training code can be built and tested before the real site is running. A WebArena environment only needs `start_url`, `prepare` and `get_state`.

## Adding a component

1. Write a class or function in the right module (or a new module).
2. Decorate it with `@ENVS.register("name")`, `@POLICIES.register(...)`, `@REWARDS.register(...)` or `@CHECKS.register(...)`.
3. If it's in a new module, add the module to `PLUGIN_MODULES` in `src/browser_agent/registry.py`.
4. Select it from a config (`env.name: name`) and add a test.
