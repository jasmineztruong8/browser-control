# Configuration

All settings live in YAML files in `experiments/configs/`. Code reads them through `browser_agent.config.load_config`, which validates every key. A misspelled or unknown key stops the run with a message naming it.

## How configs combine

1. **Defaults**: `base.yaml` lists every key with its default.
2. **Experiment files** start with `inherits: <file>` and only list what they change. Chains are allowed: `grpo_subgoal.yaml` → `grpo_binary.yaml` → `rloo.yaml` → `base.yaml`.
3. **Command-line overrides** come last, as `--set section.key=value`. Values are parsed as YAML, so `5`, `true`, `null` and `[a, b]` get the right types.
4. **Environment variables**: any string value can use `${NAME}` or `${NAME:-default}`. Values are read from your shell or from a `.env` file in the repo root (copy `.env.example`). Use these for machine-specific paths, URLs and tokens, which should never be committed.

```bash
python -m browser_agent.rollout --config experiments/configs/grpo_subgoal.yaml \
    --set env.max_steps=15 tasks.split=val run.name=quick_check
```

Each run writes its fully resolved config to `runs/<run_id>/config.yaml`, so any result can be traced back to its exact settings.

## Shipped configs

| File | Purpose |
| --- | --- |
| `base.yaml` | Defaults: mock shop, scripted policy, binary reward |
| `sft.yaml` | Condition A: SFT on verified demonstrations |
| `rloo.yaml` | Condition B: RLOO from the SFT checkpoint, binary reward |
| `grpo_binary.yaml` | Condition C: GRPO, same binary reward as B |
| `grpo_subgoal.yaml` | Condition D: GRPO with bounded subgoal rewards |
| `eval_plan_then_act.yaml` | Reasoning comparison: plan-then-act prompts with the same limits |

## Keys

### `run`

| Key | Default | Description |
| --- | --- | --- |
| `name` | `default` | Appended to the run id and output folder name |
| `output_dir` | `${RUNS_DIR:-runs}` | Where run folders are written |
| `seed` | `0` | Random seed |
| `save_prompts` | `true` | Store each step's prompt messages in trajectories (needed for SFT/RL; turn off to save disk) |

### `env`

| Key | Default | Description |
| --- | --- | --- |
| `name` | `mock_shop` | Registered environment to use |
| `start_url` | `null` | Site URL. `null` uses the built-in mock shop page |
| `headless` | `true` | Run Chromium without a window. Set `false` to watch the agent |
| `max_steps` | `12` | Maximum actions per episode |
| `action_timeout_ms` | `5000` | Timeout for a single click or type |
| `navigation_timeout_ms` | `15000` | Timeout for page loads |
| `settle_ms` | `100` | Extra wait after each action before observing |
| `scroll_pixels` | `600` | Distance moved by one scroll action |
| `viewport.width`, `viewport.height` | `1280`, `800` | Browser window size |

### `observation`

| Key | Default | Description |
| --- | --- | --- |
| `max_elements` | `60` | Maximum actionable elements listed per page |
| `max_label_chars` | `80` | Truncation length for element labels and values |
| `max_options` | `10` | Maximum dropdown options listed per element |
| `include_page_text` | `true` | Include visible page text in the observation |
| `max_page_text_chars` | `600` | Truncation length for page text |
| `max_errors` | `5` | Maximum form/validation errors shown |

### `agent`

| Key | Default | Description |
| --- | --- | --- |
| `allowed_actions` | `[click, type, scroll, stop]` | Actions the model may use. Others count as invalid |
| `history_length` | `3` | Number of previous actions shown in the prompt |
| `max_consecutive_invalid` | `3` | End the episode after this many invalid or failed actions in a row |

### `prompt`

| Key | Default | Description |
| --- | --- | --- |
| `mode` | `action_only` | `action_only` or `plan_then_act` |
| `template_dir` | `null` | Folder with your own copies of the prompt templates. `null` uses `src/browser_agent/prompts/` |

### `policy`

| Key | Default | Description |
| --- | --- | --- |
| `name` | `scripted` | `scripted` (replays task demos) or `hf` (Hugging Face model) |
| `model_name` | `Qwen/Qwen2.5-3B-Instruct` | Base model for `hf` |
| `adapter_path` | `null` | LoRA adapter to load on top of the base model |
| `load_in_4bit` | `true` | 4-bit (NF4) quantized loading |
| `compute_dtype` | `bfloat16` | `bfloat16`, `float16` or `float32` |
| `device_map` | `auto` | Passed to `from_pretrained` |
| `max_new_tokens` | `96` | Generation limit per step. Keep it equal across prompt modes |
| `temperature` | `0.0` | `0` = greedy decoding. RL configs use sampling |
| `top_p` | `1.0` | Nucleus sampling when `temperature > 0` |

### `reward`

| Key | Default | Description |
| --- | --- | --- |
| `type` | `binary` | `binary` or `subgoal` |
| `success_reward` | `1.0` | Reward for task success |
| `subgoal_bonus` | `0.1` | Bonus per verified subgoal (`subgoal` only). Each subgoal counts once |
| `subgoal_cap` | `0.3` | Maximum total bonus. Must be less than `success_reward` |
| `subgoal_normalize` | `false` | If true, bonus = `subgoal_cap` × fraction of the task's subgoals hit (ignores `subgoal_bonus`), so every task family has the same maximum bonus. On in `grpo_subgoal.yaml` |
| `require_stop` | `true` | Success also requires the agent to issue `stop` |

### `tasks`

| Key | Default | Description |
| --- | --- | --- |
| `path` | `data/tasks/mock_shop.json` | Task file. The 8-task sample is the default, for the quickstart and tests. `sft.yaml`, `rloo.yaml` (and the GRPO configs that inherit it) and `eval_plan_then_act.yaml` use the 108 generated tasks in `mock_shop_v1.json` |
| `split` | `train` | `train`, `val`, `test` or `null` for all |
| `families` | `null` | Only these workflow families, e.g. `[add_to_cart]` |
| `limit` | `null` | Only the first N matching tasks |

### `training`

Shared by the SFT and RL trainers.

| Key | Default | Description |
| --- | --- | --- |
| `method` | `none` | `sft`, `rloo`, `grpo` or `none` |
| `init_checkpoint` | `null` | Adapter to start from. All RL conditions use the same SFT checkpoint |
| `output_dir` | `${CHECKPOINT_DIR:-checkpoints}` | Where adapters are saved |
| `learning_rate` | `1e-5` | Optimizer learning rate |
| `epochs` | `1` | Passes over the data (SFT) |
| `batch_size` | `4` | Per-device batch size |
| `grad_accum_steps` | `4` | Gradient accumulation steps |
| `group_size` | `4` | Rollouts per task for the RLOO / GRPO baseline |
| `max_env_interactions` | `null` | Browser-step budget, matched across RL conditions |
| `kl_coef` | `0.0` | KL penalty to the reference policy |
| `gradient_checkpointing` | `true` | Trade compute for memory |
| `warmup_ratio` | `0.1` | Fraction of optimizer steps with linear LR warmup; the LR then decays linearly to 0 |
| `max_grad_norm` | `1.0` | Gradient clipping norm. `null` turns clipping off |
| `demos_path` | `null` | SFT: a `trajectories.jsonl` of demonstrations. `null` collects them with the scripted policy on the `train` and `val` splits of `tasks.path` |
| `val_rollouts` | `true` | SFT: after each epoch, run greedy episodes on the `val` split and keep the checkpoint with the highest success (ties go to lower validation loss). `false` selects by validation loss only |
| `lora.r`, `lora.alpha`, `lora.dropout`, `lora.target_modules` | `16`, `32`, `0.05`, `[q_proj, k_proj, v_proj, o_proj]` | LoRA adapter settings |

## Adding a new key

1. Add the field with a default to the matching section class in `src/browser_agent/config.py`.
2. Add it to `base.yaml` and to the table above.
3. Read it from `cfg.<section>.<key>` in code. Don't hard-code the value.
