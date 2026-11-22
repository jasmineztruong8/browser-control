"""Policies map a prompt to the model's raw text output. Selected by ``policy.name``.

- ``scripted``: replays a task's ``demo`` steps, resolving targets by label on the live page.
  Used for tests and for generating verified demonstrations for SFT.
- ``hf``: a Hugging Face causal LM (optionally 4-bit, optionally with a LoRA adapter).
  Requires ``pip install -e ".[train]"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from browser_agent.config import Config
from browser_agent.registry import POLICIES
from browser_agent.tasks import DemoStep, Task
from browser_agent.types import Action, ActionKind, Element, Observation, StepRecord


@dataclass
class PolicyContext:
    task: Task
    observation: Observation
    step_index: int
    history: list[StepRecord] = field(default_factory=list)


@dataclass
class PolicyOutput:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Policy(ABC):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def act(self, messages: list[dict[str, str]], ctx: PolicyContext) -> PolicyOutput:
        """Return the raw model output for this turn; the rollout parses it into an action."""

    def close(self) -> None:
        pass


def find_element(elements: list[Element], target: str, role: str | None = None) -> Element | None:
    """Exact (case-insensitive) label match first, then substring match; optional role filter."""
    candidates = [e for e in elements if role is None or e.role == role]
    t = target.lower()
    for match in (lambda e: e.label.lower() == t, lambda e: t in e.label.lower()):
        for e in candidates:
            if match(e):
                return e
    return None


@POLICIES.register("scripted")
class ScriptedPolicy(Policy):
    def act(self, messages: list[dict[str, str]], ctx: PolicyContext) -> PolicyOutput:
        demo = ctx.task.demo
        if not demo:
            raise ValueError(f"task {ctx.task.id} has no 'demo' for the scripted policy")
        if ctx.step_index >= len(demo):
            return PolicyOutput(text="stop")
        return PolicyOutput(text=self._render(demo[ctx.step_index], ctx.observation))

    @staticmethod
    def _render(step: DemoStep, obs: Observation) -> str:
        kind = ActionKind(step.action)
        if kind is ActionKind.STOP:
            return Action(kind=kind, answer=step.answer).to_text()
        if kind is ActionKind.SCROLL:
            return Action(kind=kind, direction=step.direction or "down").to_text()
        element = find_element(obs.elements, step.target or "", step.role)
        if element is None:
            # Unparseable on purpose: the rollout records it as an invalid action.
            return f"[scripted] target not found: {step.target!r}"
        return Action(
            kind=kind, element_id=element.id, text=step.text, submit=step.submit
        ).to_text()


@POLICIES.register("hf")
class HFPolicy(Policy):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'The "hf" policy needs the training extras: pip install -e ".[train]"'
            ) from exc

        p = cfg.policy
        self._torch = torch
        dtype = getattr(torch, p.compute_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(p.model_name)
        kwargs: dict = {"device_map": p.device_map, "torch_dtype": dtype}
        if p.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        self.model = AutoModelForCausalLM.from_pretrained(p.model_name, **kwargs)
        if p.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, p.adapter_path)
        self.model.eval()

    def act(self, messages: list[dict[str, str]], ctx: PolicyContext) -> PolicyOutput:
        p = self.cfg.policy
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        gen: dict = {
            "max_new_tokens": p.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if p.temperature > 0:
            gen.update(do_sample=True, temperature=p.temperature, top_p=p.top_p)
        else:
            gen["do_sample"] = False
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0, prompt_len:]
        return PolicyOutput(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            prompt_tokens=int(prompt_len),
            completion_tokens=int(new_tokens.shape[0]),
        )
