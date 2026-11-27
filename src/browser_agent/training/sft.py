"""Condition A: supervised fine-tuning of a LoRA adapter on verified demonstrations.

Command line (on the GPU machine)::

    python -m browser_agent.training.sft --config experiments/configs/sft.yaml

What it does:

1. Collects demonstrations by replaying each task's scripted ``demo`` on the ``train`` and
   ``val`` splits, keeping only successful episodes (or reads ``training.demos_path``).
2. Turns every valid demo step into a ``(prompt messages -> action text)`` example. The loss
   counts only the action tokens.
3. Trains a LoRA adapter for ``training.epochs`` epochs. After each epoch it measures the
   validation loss and, with ``training.val_rollouts``, the greedy success rate on the ``val``
   tasks in the live environment. Epoch 0 is the untrained base model, for reference.
4. Keeps the epoch with the best validation success (ties: lower validation loss) and copies
   it to ``<training.output_dir>/<run.name>``, i.e. ``checkpoints/sft``, where the RL configs
   expect it.

Writes ``<run.output_dir>/<timestamp>-<run.name>/`` with ``config.yaml``, ``demos.jsonl``,
``metrics.jsonl`` (one line per optimizer step and per evaluation), ``val_epoch<k>.jsonl``
(validation episodes) and ``summary.json``. Every epoch's adapter is kept under
``<training.output_dir>/<run_id>/epoch-<k>``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from browser_agent.config import Config, ConfigError, load_config, save_config
from browser_agent.env.base import BrowserEnv
from browser_agent.rollout import make_env, run_episode, summarize
from browser_agent.tasks import load_tasks
from browser_agent.training.data import (
    SFTExample,
    collect_demos,
    examples_from_trajectories,
    load_demos,
)
from browser_agent.types import Trajectory, save_jsonl

DEFAULT_CONFIG = "experiments/configs/sft.yaml"


def _log(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _demos(env: BrowserEnv, cfg: Config, run_id: str) -> dict[str, list[Trajectory]]:
    if cfg.training.demos_path:
        by_split = load_demos(cfg.training.demos_path)
        print(f"Loaded demos from {cfg.training.demos_path}")
        return {"train": by_split.get("train", []), "val": by_split.get("val", [])}
    return {split: collect_demos(env, cfg, split, run_id) for split in ("train", "val")}


def _report_demos(demos: dict[str, list[Trajectory]]) -> None:
    for split, trajs in demos.items():
        ok = sum(t.success for t in trajs)
        print(f"{split}: {ok}/{len(trajs)} demos succeeded")
        for t in trajs:
            if not t.success:
                print(f"  dropped failed demo {t.task_id} (end={t.terminated})")


def train(cfg: Config, run_dir: Path, run_id: str) -> dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    from browser_agent.training.common import (
        IGNORE_INDEX,
        collate,
        encode,
        generation_mode,
        load_trainable_policy,
        training_mode,
    )

    t = cfg.training
    set_seed(cfg.run.seed)
    metrics = run_dir / "metrics.jsonl"

    with make_env(cfg) as env:
        demos = _demos(env, cfg, run_id)
        _report_demos(demos)
        save_jsonl(demos["train"] + demos["val"], run_dir / "demos.jsonl")
        train_examples = examples_from_trajectories(demos["train"])
        val_examples = examples_from_trajectories(demos["val"])
        if not train_examples:
            raise RuntimeError("no successful train demo steps to train on")
        print(f"{len(train_examples)} train examples, {len(val_examples)} val examples")

        start = time.perf_counter()
        policy = load_trainable_policy(cfg)
        model, tok = policy.model, policy.tokenizer
        print(f"model loaded in {time.perf_counter() - start:.1f}s")
        device = model.device
        pad_id = tok.pad_token_id

        def encode_all(examples: list[SFTExample]):
            return [encode(tok, ex.messages, ex.target) for ex in examples]

        train_enc, val_enc = encode_all(train_examples), encode_all(val_examples)
        lengths = [len(ids) for ids, _ in train_enc]
        token_stats = {"mean": round(statistics.mean(lengths)), "max": max(lengths)}
        print(f"sequence tokens: {token_stats}")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=t.learning_rate)
        steps_per_epoch = math.ceil(len(train_enc) / (t.batch_size * t.grad_accum_steps))
        total_steps = steps_per_epoch * t.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * t.warmup_ratio), total_steps
        )

        val_cfg = cfg.model_copy(deep=True)
        val_cfg.tasks.split = "val"  # type: ignore[assignment]
        vt = val_cfg.tasks
        val_tasks = load_tasks(vt.path, vt.split, vt.families, vt.limit)

        def val_loss() -> float | None:
            if not val_enc:
                return None
            model.eval()
            total, tokens = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_enc), t.batch_size):
                    batch = collate(val_enc[i : i + t.batch_size], pad_id, device)
                    n = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
                    total += model(**batch).loss.item() * n
                    tokens += n
            return total / tokens

        def evaluate(epoch: int) -> dict[str, Any]:
            record: dict[str, Any] = {"type": "eval", "epoch": epoch, "val_loss": val_loss()}
            if t.val_rollouts and val_tasks:
                generation_mode(model)
                start = time.perf_counter()
                trajs = [run_episode(env, policy, task, val_cfg, run_id) for task in val_tasks]
                save_jsonl(trajs, run_dir / f"val_epoch{epoch}.jsonl")
                summary = summarize(trajs)
                record.update({f"val_{k}": v for k, v in summary.items()})
                record["val_rollout_s"] = round(time.perf_counter() - start, 1)
            _log(metrics, record)
            print(" ".join(f"{k}={_fmt(v)}" for k, v in record.items()))
            return record

        print(
            f"{t.epochs} epochs x {steps_per_epoch} optimizer steps "
            f"(batch {t.batch_size} x accum {t.grad_accum_steps})"
        )
        evals = [evaluate(0)]  # the base model, for reference; never selected
        rng = random.Random(cfg.run.seed)
        order = list(range(len(train_enc)))
        step = 0
        train_start = time.perf_counter()
        for epoch in range(1, t.epochs + 1):
            training_mode(model)
            rng.shuffle(order)
            micro = [order[i : i + t.batch_size] for i in range(0, len(order), t.batch_size)]
            losses: list[float] = []
            for m, idx in enumerate(micro):
                batch = collate([train_enc[i] for i in idx], pad_id, device)
                # The last accumulation window of an epoch may be short; scale by its real size.
                window_start = m - m % t.grad_accum_steps
                window = min(t.grad_accum_steps, len(micro) - window_start)
                loss = model(**batch).loss
                (loss / window).backward()
                losses.append(loss.item())
                if m - window_start + 1 == window:
                    grad_norm = None
                    if t.max_grad_norm:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    record = {
                        "type": "train",
                        "epoch": epoch,
                        "step": step,
                        "loss": statistics.mean(losses[-window:]),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                    }
                    _log(metrics, record)
                    if step % 5 == 0 or step == total_steps:
                        print(" ".join(f"{k}={_fmt(v)}" for k, v in record.items()))
            adapter = Path(t.output_dir) / run_id / f"epoch-{epoch}"
            model.save_pretrained(adapter)
            record = evaluate(epoch)
            record["adapter"] = str(adapter)
            record["train_loss"] = statistics.mean(losses)
            evals.append(record)
        train_seconds = time.perf_counter() - train_start
        policy.close()

    best = max(evals[1:], key=_selection_key)
    final = Path(t.output_dir) / cfg.run.name
    if final.exists():
        print(f"replacing the previous adapter at {final}")
        shutil.rmtree(final)
    shutil.copytree(best["adapter"], final)
    selection = {"run_id": run_id, "selected": best, "evals": evals}
    (final / "selection.json").write_text(json.dumps(selection, indent=2))

    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return {
        "run_id": run_id,
        "adapter": str(final),
        "selected_epoch": best["epoch"],
        "selected": {k: v for k, v in best.items() if k.startswith("val_")},
        "base": {k: v for k, v in evals[0].items() if k.startswith("val_")},
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "sequence_tokens": token_stats,
        "optimizer_steps": step,
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_memory_gb": peak_gb,
    }


def _selection_key(record: dict[str, Any]) -> tuple[float, float]:
    success = record.get("val_success_rate")
    loss = record.get("val_loss")
    return (
        success if success is not None else -math.inf,
        -loss if loss is not None else -math.inf,
    )


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the SFT baseline adapter.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if cfg.training.method != "sft":
        print(f"training.method is {cfg.training.method!r}, expected 'sft'", file=sys.stderr)
        return 2
    if cfg.policy.name != "hf":
        print("policy.name must be 'hf' to train a model", file=sys.stderr)
        return 2

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.run.name}"
    run_dir = Path(cfg.run.output_dir) / run_id
    save_config(cfg, run_dir / "config.yaml")

    summary = train(cfg, run_dir, run_id)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved run to {run_dir}; adapter at {summary['adapter']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
