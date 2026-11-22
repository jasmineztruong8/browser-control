"""Turn a live page into a compact ``Observation`` and render it as prompt text.

Extraction tags each actionable element with a ``data-agent-id`` attribute so the environment
can act on it by id. Ids are re-assigned on every observation (in DOM order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from browser_agent.config import ObservationConfig
from browser_agent.types import Element, Observation

if TYPE_CHECKING:
    from playwright.sync_api import Page

AGENT_ID_ATTR = "data-agent-id"

_EXTRACT_JS = """
(opts) => {
  const ATTR = opts.attr;
  document.querySelectorAll(`[${ATTR}]`).forEach(e => e.removeAttribute(ATTR));
  const selector = [
    'a[href]', 'button', 'input:not([type=hidden])', 'select', 'textarea',
    '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]', '[role=tab]',
    '[role=menuitem]', '[role=option]', '[role=combobox]', '[role=textbox]'
  ].join(',');
  const roleOf = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (['checkbox', 'radio'].includes(t)) return t;
      if (['submit', 'button', 'reset'].includes(t)) return 'button';
      if (t === 'number') return 'spinbutton';
      if (t === 'search') return 'searchbox';
      return 'textbox';
    }
    return tag;
  };
  const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  const labelOf = (el) => clean(
    el.getAttribute('aria-label') ||
    (el.labels && el.labels.length ? el.labels[0].innerText : '') ||
    el.getAttribute('placeholder') ||
    el.innerText ||
    (el.tagName === 'INPUT' ? el.value : '') ||
    el.getAttribute('title') || el.getAttribute('name') || ''
  );
  const visible = (el) => {
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 || r.height > 0;
  };

  const elements = [];
  for (const el of document.querySelectorAll(selector)) {
    if (elements.length >= opts.maxElements) break;
    if (el.disabled || !visible(el)) continue;
    const id = elements.length;
    el.setAttribute(ATTR, String(id));
    const role = roleOf(el);
    let value = null, options = null;
    if (el.tagName === 'SELECT') {
      const sel = el.options[el.selectedIndex];
      value = sel && sel.value ? clean(sel.text) : null;
      options = Array.from(el.options).filter(o => o.value).map(o => clean(o.text));
    } else if (['INPUT', 'TEXTAREA'].includes(el.tagName)) {
      const t = (el.getAttribute('type') || '').toLowerCase();
      value = ['checkbox', 'radio'].includes(t) ? String(el.checked) : (el.value || null);
    }
    elements.push({id, role, label: labelOf(el), value, options});
  }

  const errors = [];
  for (const el of document.querySelectorAll('[role=alert], [aria-invalid=true], .error')) {
    const text = clean(el.innerText || el.getAttribute('aria-errormessage'));
    if (text && !errors.includes(text)) errors.push(text);
  }
  return {
    title: document.title,
    elements,
    errors,
    page_text: opts.includeText ? clean(document.body ? document.body.innerText : '') : '',
  };
}
"""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def observe(page: Page, cfg: ObservationConfig) -> Observation:
    raw = page.evaluate(
        _EXTRACT_JS,
        {
            "attr": AGENT_ID_ATTR,
            "maxElements": cfg.max_elements,
            "includeText": cfg.include_page_text,
        },
    )
    elements = []
    for e in raw["elements"]:
        options = e.get("options")
        if options is not None:
            options = [_truncate(o, cfg.max_label_chars) for o in options[: cfg.max_options]]
        value = e.get("value")
        elements.append(
            Element(
                id=e["id"],
                role=e["role"],
                label=_truncate(e["label"], cfg.max_label_chars),
                value=_truncate(value, cfg.max_label_chars) if value else None,
                options=options,
            )
        )
    return Observation(
        url=page.url,
        title=raw["title"],
        elements=elements,
        page_text=_truncate(raw["page_text"], cfg.max_page_text_chars),
        errors=raw["errors"][: cfg.max_errors],
    )


def format_observation(obs: Observation) -> str:
    """Render an observation as the compact text the model sees."""
    lines = [f"URL: {obs.url}"]
    if obs.title:
        lines.append(f"Title: {obs.title}")
    if obs.errors:
        lines.append("Errors: " + " | ".join(obs.errors))
    lines.append("Elements:")
    for e in obs.elements:
        line = f'[{e.id}] {e.role} "{e.label}"'
        if e.value:
            line += f' value="{e.value}"'
        if e.options:
            line += " options=[" + ", ".join(e.options) + "]"
        lines.append(line)
    if obs.page_text:
        lines.append(f"Page text: {obs.page_text}")
    return "\n".join(lines)
