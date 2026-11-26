# Compute pilot (GCP, NVIDIA L4)

The pilot measures what the training runs cost before we launch them. It runs `notebooks/compute_pilot.ipynb` once on the team VM (`scripts/gcp/vm.sh pilot`, 2026-11-26). The files here are its outputs:

- `summary.md`: the measurement table
- `pilot.json`: every number
- `config.yaml`: the resolved config

## Setup

| | |
| --- | --- |
| Hardware | g2-standard-8 + 1× NVIDIA L4 (22 GB), Spot, `us-central1-a` |
| Model | Qwen2.5-3B-Instruct, 4-bit NF4, bf16 compute, LoRA r=16 on q/k/v/o (7.4M trainable parameters) |
| Config | `experiments/configs/rloo.yaml`: action-only prompts, `max_steps=12`, group size 4 |
| Tasks | The 8-task sample `data/tasks/mock_shop.json`. The branch predates the switch to `mock_shop_v1.json`. Prompt lengths are the same either way |

## Findings

- **Memory isn't a constraint.** Peak use was 5.7 GB of 22 GB, during an SFT step with batch 4. A larger SFT batch, a larger RL group, or a 7B model in 4-bit would all fit.
- **Generation dominates.** Each environment step takes 0.73 s of model generation plus 0.16 s of browser time. Each sequence in the RL update adds 0.55 s. That is about 1.44 s per interaction, or about **2,500 RL interactions per GPU hour**.
- **SFT is cheap.** It takes 0.3 s per example, so SFT on a few hundred demonstrations takes minutes.
- **Saved adapters reload exactly.** The LoRA weights are identical after a reload (28 MB). Logits differ by about 1.1, because training runs layer norms and the output layer in fp32 while inference runs them in bf16. The top token is unchanged.

## RL budget

At 2,500 interactions per GPU hour, the RL conditions B (RLOO), C (GRPO) and D (GRPO with subgoal rewards) cost:

| Interactions per condition | Per condition | B + C + D |
| --- | --- | --- |
| 1,000 | 0.4 h | 1.2 h |
| 2,500 | 1.0 h | 3.0 h |
| 5,000 | 2.0 h | 6.0 h |
| 10,000 | 4.0 h | 12.0 h |

## Caveats

These are lower bounds:

- **Small sample.** Throughput comes from 4 rollouts (20 steps) and one update. Generation was one episode at a time, with about 6 completion tokens per action.
- **Plan-then-act is slower.** It writes a plan before each action, so its generation time grows with plan length.
- **Overheads aren't counted.** Periodic validation, checkpointing and model loading add time on top of these figures.

Plan for about 25% headroom.
