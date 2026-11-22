"""Parse model output into a structured ``Action``.

Action grammar (one action per response)::

    click [12]
    type [5] "blue mug"          # add `enter` at the end to press Enter afterwards
    scroll down                  # or: scroll up
    stop                         # or: stop "final answer"

In ``plan_then_act`` mode the model writes ``Plan: ...`` then ``Action: <action>``; the text
after the last ``Action:`` is parsed. Otherwise the last non-empty line is parsed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from browser_agent.types import Action, ActionKind


class InvalidActionError(ValueError):
    pass


_PATTERNS = {
    ActionKind.CLICK: re.compile(r"click\s*\[(\d+)\]", re.I),
    ActionKind.TYPE: re.compile(r'type\s*\[(\d+)\]\s*"((?:[^"\\]|\\.)*)"(\s+enter)?', re.I),
    ActionKind.SCROLL: re.compile(r"scroll\s+(up|down)", re.I),
    ActionKind.STOP: re.compile(r'stop(?:\s+"((?:[^"\\]|\\.)*)")?', re.I),
}
_ACTION_MARKER = re.compile(r"action\s*:", re.I)


def extract_action_text(output: str) -> str:
    text = output.replace("```", "\n").strip()
    markers = list(_ACTION_MARKER.finditer(text))
    if markers:
        text = text[markers[-1].end() :]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        candidate = lines[0] if lines else ""
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        candidate = lines[-1] if lines else ""
    return candidate.strip("`*- ").rstrip(".").strip()


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\")


def parse_action(output: str, allowed: Iterable[str] | None = None) -> Action:
    candidate = extract_action_text(output)
    if not candidate:
        raise InvalidActionError("empty response")

    for kind, pattern in _PATTERNS.items():
        m = pattern.fullmatch(candidate)
        if not m:
            continue
        if allowed is not None and kind.value not in set(allowed):
            raise InvalidActionError(f"action '{kind.value}' is not allowed")
        if kind is ActionKind.CLICK:
            return Action(kind=kind, element_id=int(m.group(1)))
        if kind is ActionKind.TYPE:
            return Action(
                kind=kind,
                element_id=int(m.group(1)),
                text=_unescape(m.group(2)),
                submit=bool(m.group(3)),
            )
        if kind is ActionKind.SCROLL:
            return Action(kind=kind, direction=m.group(1).lower())
        answer = m.group(1)
        return Action(kind=kind, answer=_unescape(answer) if answer is not None else None)

    raise InvalidActionError(f"could not parse action from: {candidate[:120]!r}")
