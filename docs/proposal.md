# Final Project Proposal

**Team:** Gia Yang (cy2866), Jasmine Truong (jt3595), Nicolas Hunt (2905)

## Overview

We propose investigating how reinforcement learning (RL) rewards, policy optimization, and reasoning affect browser-control agents for users completing repetitive, multi-step web tasks, such as selecting products and updating a shopping cart.

## Environment and Model

- **Model:** Qwen2.5-3B-Instruct (provisional)
- **Environment:** One self-hosted WebArena shopping site [1]
- **Tasks:** 2–3 short workflow families
- **Action space:** `click`, `type`, and `scroll`

### Observation Processing

To reduce context and compute costs, we will first extract compact DOM/accessibility text containing:

- Actionable elements
- Relevant page state
- Form errors

Irrelevant content will be removed and history will be limited.

## Method

Building on WebAgent-R1 [2] and WebRL [3]:

1. **SFT baseline:** Establish a supervised fine-tuning (SFT) baseline from verified demonstrations.
2. **Policy optimization comparison:** Compare REINFORCE with a leave-one-out baseline (RLOO) [4] and Group Relative Policy Optimization (GRPO) [5] using identical binary task-success rewards.
3. **Reward design comparison:** Holding GRPO fixed, compare binary rewards with bounded, evaluator-verified subgoal rewards. Each subgoal will count once and final success will dominate.
   - *Optional extension:* an efficiency-based reward.
4. **Reasoning comparison:** Compare action-only prompting with a brief plan before acting, under matched token and action limits.

All RL runs will start from the same SFT checkpoint.

## Experimental Controls

- Separate training, validation, and held-out task instances
- Controlled observations, tools, initial states, and training interaction budgets

## Evaluation

We will analyze:

- Task completion
- Recovery
- Invalid actions
- Learning curves
- Latency
- GPU hours

Repeated runs and uncertainty estimates will be included as resources permit.

## Compute Feasibility

Compute feasibility will be assessed through an early pilot on the allocated course GPU, measuring peak memory and rollout throughput before fixing the experiment budget.

Efficiency techniques:

- LoRA
- Gradient checkpointing
- Compatible 4-bit loading
- Short contexts
- Small rollout groups

If necessary, we will reduce task size or use a smaller Qwen checkpoint consistently.

## Timeline

| Period | Focus |
| --- | --- |
| Late October | Setup and profiling |
| November | Training and analysis |
| Early December | Demo |

## Deliverables

- Model adapters
- A runnable browser agent
- Reproducible experiments
- Final report

## References

1. Zhou et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024.
2. Wei et al. *WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning.* EMNLP 2025.
3. Qi et al. *WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning.* ICLR 2025.
4. Ahmadian et al. *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.* 2024.
5. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024.
