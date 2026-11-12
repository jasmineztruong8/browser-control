# Browser-Control Agent: Project Roadmap and Team Responsibilities

## Project Goal

Develop a browser-control agent for one self-hosted shopping website and investigate how policy-optimization methods, reward design, and reasoning prompts affect task completion. Establish a supervised fine-tuning (SFT) baseline, then compare reinforcement learning (RL) approaches under consistent evaluation conditions.

## Scope

- One website with two or three short workflows, such as finding a product, selecting its options, and updating a cart.
- A small instruction-tuned model, provisionally Qwen2.5-3B-Instruct, subject to compute testing.
- Compact DOM/accessibility observations and structured browser actions.
- Four main experimental conditions, with recovery-focused training as an optional extension.
- Planning assumption: approximately 8–10 active hours per person per week, plus unattended training jobs.

## Team Responsibilities

### Gia: Environment, tasks, and rewards

- Deploy the website and implement reliable environment resets.
- Create task instances and train/validation/test splits.
- Implement automatic task-success and intermediate-progress checks.
- Lead the reward-design experiment.
- Write the environment, dataset, and reward-design sections.

### Jasmine: Model training and compute

- Confirm the model checkpoint, GPU availability, and compute budget.
- Prepare demonstrations and train the SFT baseline.
- Implement the basic RLOO training condition.
- Track memory usage, training stability, and GPU hours.
- Write the training-methods and compute-analysis sections.

### Nicolas: Browser integration, evaluation, and demo

- Implement DOM extraction, action execution, and trajectory logging.
- Integrate GRPO with the shared browser rollout pipeline.
- Build the evaluation harness and compare reasoning prompts.
- Coordinate the working demo and results figures.
- Write the evaluation, results, and reproduction instructions.

Everyone will review experimental decisions, run at least one training or evaluation job, contribute to the presentation, and review another teammate's report section.

## Experiment Plan

- **Condition A: SFT baseline.** Train on verified demonstrations and measure initial task performance.
- **Condition B: SFT + RLOO.** Introduce a basic REINFORCE-style RL baseline using binary task-success rewards.
- **Condition C: SFT + GRPO.** Keep the same binary reward to isolate the effect of the optimization method.
- **Condition D: SFT + GRPO with reward shaping.** Add bounded rewards for verified intermediate progress to isolate the effect of reward design.
- **Reasoning-prompt comparison.** Compare action-only responses with brief planning before actions using the same selected checkpoint and shared token/action limits.

Every RL condition will start independently from the same SFT checkpoint. Training checkpoints will not be chained from one experimental condition to the next.

## Schedule

### November 12–15: Environment Setup and Compute Pilot

- **Gia:** Deploy the website, implement resets, and define correct outcomes for each workflow. Test whether success checks reject incorrect products, variants, quantities, and false completion claims.
- **Jasmine:** Confirm GPU specifications and credits. Load the selected model and adapter configuration, then measure memory and runtime for a small training update.
- **Nicolas:** Connect browser observations to a fixed DOM/action format. Implement `click`, `type`, `scroll`, and `stop` actions, along with timeouts and episode logs.
- **Together:** Agree on the shared trajectory format and verify that both RL implementations can use it.

**Milestone:** Complete one browser episode, calculate its reward, reset the environment, perform a real RL update, and save/reload the adapter. Use these measurements to confirm or reduce the planned scope.

### November 16–20: Demonstrations and SFT Baseline

- **Gia:** Create versioned task splits and collect approximately 50–100 verified short demonstrations using human actions or parameterized scripts.
- **Jasmine:** Convert demonstrations into observation/action training examples, train the SFT adapter, and select a checkpoint using validation tasks.
- **Nicolas:** Finish the reusable evaluation runner and build a basic text-input demo. Log actions, outcomes, tokens, latency, and model configuration.

**Milestone:** A saved SFT checkpoint, reproducible baseline results, and a working evaluation command. Freeze the main observation and action formats before comparing RL methods.

### November 21–24: Basic RL Baseline

- **Jasmine:** Train Condition B using RLOO and binary success rewards.
- **Gia:** Audit reward correctness and examine whether training tasks provide a useful mix of successful and unsuccessful attempts.
- **Nicolas:** Review trajectory handling with Jasmine and test GRPO on a small pilot using the same interface.

**Milestone:** A reproducible RLOO run with learning curves, saved weights, and measured compute usage. Resolve training, reset, and evaluator errors before larger comparisons.

### November 25–28: Policy-Optimization Comparison

- **Nicolas:** Train Condition C using GRPO and the same binary reward as Condition B.
- **Jasmine:** Verify shared initialization, comparable training-interaction budgets, and consistent evaluation settings.
- **Gia:** Implement and test intermediate-progress rewards for the next experiment, ensuring repeated actions cannot repeatedly earn the same bonus.
- **Together:** Schedule long-running jobs around Thanksgiving availability.

**Milestone:** A controlled RLOO-versus-GRPO comparison covering task success, training stability, interaction efficiency, and compute cost.

### November 29–December 2: Reward and Reasoning Experiments

- **Gia:** Train Condition D using binary success plus bounded bonuses for verified subgoals.
- **Jasmine:** Review reward behavior and repeat the most important comparison with another seed if compute permits.
- **Nicolas:** Compare action-only and brief-plan prompts. Freeze configurations, evaluate Conditions A–D on held-out tasks, and summarize failures and uncertainty.

**Milestone:** Main experiments and test results complete by December 2.

### December 3–6: Demo, Analysis, and Presentation

- **Gia:** Verify clean website startup and repeatable resets. Document environment limitations and review references.
- **Jasmine:** Finalize training curves, compute tables, and explanations of the RL methods.
- **Nicolas:** Stabilize the demo, prepare results figures, and record a backup demonstration.
- **Together:** Draft the report, prepare slides, and rehearse the presentation.

**Milestone:** Freeze the demo by December 5. Have a report draft and setup instructions ready, with another teammate successfully launching the project from the README.

### December 7–13: Presentation and Final Documentation

- Present and demonstrate the system in the assigned slot. December 7 is the earliest planned presentation date; confirm the team's actual slot.
- Finish report sections, explain negative or inconclusive results, and document limitations.
- Clean the repository and include configurations, evaluation scripts, results, tests, and reproduction instructions.
- Cross-review each other's sections and verify that conclusions match the evidence.

**Milestone:** Presentation completed and submission materials nearly final.

### December 14–15: Final Checks and Submission

- **December 14:** Internal deadline for final figures, citations, report export, clean-install checks, and a complete demo rehearsal.
- **December 15:** Submit the report and repository through the required course channels and verify receipt.

## Evaluation and Compute Controls

- Use validation tasks for reward weights, prompts, checkpoint selection, and stopping decisions.
- Keep test tasks separate from demonstrations and tuning.
- Compare all conditions using final task success, regardless of their training reward.
- Report action count, invalid actions, latency, token usage, GPU hours, and failure categories.
- Match training browser-interaction budgets across RL conditions and report compute differences separately.
- Profile compute during the first week and reserve approximately 20% of credits for failures and reruns.
- If resources are insufficient, shorten context or episodes, or use a smaller model consistently across conditions. Complete the core comparisons before adding optional experiments.

## Final Deliverables

- A working browser-agent demo.
- SFT and RL model adapters or checkpoints.
- A reproducible repository with training and evaluation instructions.
- Experimental results comparing optimization methods, rewards, and reasoning prompts.
- A final report and presentation explaining methods, findings, limitations, and compute usage.
