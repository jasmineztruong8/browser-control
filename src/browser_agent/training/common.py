"""Model setup and tokenization shared by the SFT and RL trainers.

Needs the training extras: ``pip install -e ".[train]"``.

The trainable model lives inside an ``HFPolicy``, so rollouts during training (validation
episodes for SFT, sampled groups for RL) generate with exactly the weights being trained, through
the same code path that evaluation uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from browser_agent.config import Config
from browser_agent.policy import HFPolicy

IGNORE_INDEX = -100


def load_trainable_policy(cfg: Config) -> HFPolicy:
    """An ``HFPolicy`` whose model carries a trainable LoRA adapter.

    The adapter starts from ``training.init_checkpoint`` if set (RL starts from the SFT
    adapter), otherwise it is a fresh adapter built from ``training.lora``.
    """
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    base_cfg = cfg.model_copy(deep=True)
    base_cfg.policy.adapter_path = None  # the adapter is attached below, trainable
    policy = HFPolicy(base_cfg)

    t = cfg.training
    model = policy.model
    if cfg.policy.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t.gradient_checkpointing
        )
    elif t.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if t.init_checkpoint:
        model = PeftModel.from_pretrained(model, t.init_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=t.lora.r,
                lora_alpha=t.lora.alpha,
                lora_dropout=t.lora.dropout,
                target_modules=list(t.lora.target_modules),
                task_type="CAUSAL_LM",
            ),
        )
    policy.model = model
    if policy.tokenizer.pad_token_id is None:
        policy.tokenizer.pad_token = policy.tokenizer.eos_token
    return policy


def generation_mode(model: torch.nn.Module) -> None:
    model.eval()
    model.config.use_cache = True  # gradient checkpointing turns the KV cache off


def training_mode(model: torch.nn.Module) -> None:
    model.train()
    model.config.use_cache = False


def encode(tokenizer, messages: list[dict[str, str]], target: str) -> tuple[list[int], list[int]]:
    """Token ids of prompt + target, and labels with the prompt masked out.

    The prompt is rendered exactly as ``HFPolicy`` renders it at inference
    (``add_generation_prompt=True``). The target is the assistant turn as the chat template
    closes it, so the model learns to end its turn (Qwen: ``<|im_end|>``) and generation stops.
    """
    prompt = list(
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]
    )
    full = list(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}], tokenize=True, return_dict=True
        )["input_ids"]
    )
    assert full[: len(prompt)] == prompt, "chat template doesn't render the prompt as a prefix"
    return full, [IGNORE_INDEX] * len(prompt) + full[len(prompt) :]


def collate(
    encoded: Sequence[tuple[list[int], list[int]]], pad_id: int, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Right-pad encoded examples into a batch."""
    width = max(len(ids) for ids, _ in encoded)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for ids, labels in encoded:
        gap = width - len(ids)
        batch["input_ids"].append(ids + [pad_id] * gap)
        batch["attention_mask"].append([1] * len(ids) + [0] * gap)
        batch["labels"].append(labels + [IGNORE_INDEX] * gap)
    return {k: torch.tensor(v, device=device) for k, v in batch.items()}
