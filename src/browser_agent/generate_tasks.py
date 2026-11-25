"""Generate mock-shop task instances with deterministic train/validation/test splits.

Command line::

    python -m browser_agent.generate_tasks --out data/tasks/mock_shop_v1.json

Tasks are enumerated from the mock shop catalog (products x options x quantities) for the three
workflow families ``add_to_cart``, ``update_quantity`` and ``remove_item``. Every task carries a
scripted ``demo``, so running the ``scripted`` policy on a split produces verified
demonstrations (see data/README.md).

Splits are reproducible and designed so that the test split measures generalization rather
than memorization:

- The product variants in ``HELD_OUT`` are only ever the *target* of test tasks. They never
  appear as the target of a train or val task (they may still sit in the cart as a distractor).
- Every other task is assigned to train or val by hashing its id together with the seed, so the
  same seed always produces the same file.

The catalog below mirrors ``env/fixtures/mock_shop.html``; a test checks they stay in sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from browser_agent.tasks import Task

DEFAULT_OUT = "data/tasks/mock_shop_v1.json"
DEFAULT_SEED = 0
DEFAULT_VAL_FRACTION = 0.15


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    option_name: str
    options: tuple[str, ...]
    search: str  # query typed into the search box; must match only this product
    singular: str  # intent phrase for one item, with {option} / {option_lower}
    plural: str  # intent phrase for several items


CATALOG: tuple[Product, ...] = (
    Product(
        id="mug",
        name="Ceramic Coffee Mug",
        option_name="Color",
        options=("Blue", "Red", "Green"),
        search="mug",
        singular="{option_lower} Ceramic Coffee Mug",
        plural="{option_lower} Ceramic Coffee Mugs",
    ),
    Product(
        id="tshirt",
        name="Cotton T-Shirt",
        option_name="Size",
        options=("S", "M", "L"),
        search="t-shirt",
        singular="Cotton T-Shirt in size {option}",
        plural="Cotton T-Shirts in size {option}",
    ),
    Product(
        id="headphones",
        name="Wireless Headphones",
        option_name="Color",
        options=("Black", "White"),
        search="headphones",
        singular="pair of {option_lower} Wireless Headphones",
        plural="pairs of {option_lower} Wireless Headphones",
    ),
    Product(
        id="notebook",
        name="Spiral Notebook",
        option_name="Pack",
        options=("1-pack", "3-pack"),
        search="notebook",
        singular="Spiral Notebook {option}",
        plural="Spiral Notebook {option}s",
    ),
    Product(
        id="bottle",
        name="Steel Water Bottle",
        option_name="Size",
        options=("500 ml", "1 L"),
        search="bottle",
        singular="{option} Steel Water Bottle",
        plural="{option} Steel Water Bottles",
    ),
)

# (product_id, option) pairs reserved for the test split. One option value is held out for
# three products, so train still has to choose among options for every product.
HELD_OUT: frozenset[tuple[str, str]] = frozenset(
    {("mug", "Green"), ("tshirt", "L"), ("bottle", "500 ml")}
)

ADD_QUANTITIES = (1, 2, 3, 5)
# (initial quantity, target quantity, whether another item is also in the cart)
UPDATE_CASES = ((1, 3, False), (4, 2, True), (2, 5, True))
REMOVE_DISTRACTOR_COUNTS = (1, 2)

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


# --------------------------------------------------------------------------- helpers


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _variants() -> list[tuple[Product, str]]:
    return [(p, o) for p in CATALOG for o in p.options]


def _label(product: Product, option: str) -> str:
    """Label the mock shop prints for a cart item, e.g. ``Ceramic Coffee Mug (Blue)``."""
    return f"{product.name} ({option})"


def _item(product: Product, option: str, quantity: int) -> dict[str, Any]:
    return {"product_id": product.id, "option": option, "quantity": quantity}


def _phrase(product: Product, option: str, quantity: int) -> str:
    template = product.singular if quantity == 1 else product.plural
    return template.format(option=option, option_lower=option.lower())


def _distractors(
    rng: random.Random, product: Product, count: int
) -> list[tuple[Product, str, int]]:
    """Pick ``count`` items from other products that are not held out, with quantity 1 or 2."""
    pool = [(p, o) for p, o in _variants() if p.id != product.id and (p.id, o) not in HELD_OUT]
    chosen = rng.sample(pool, count)
    return [(p, o, rng.choice((1, 2))) for p, o in chosen]


def _split(task_id: str, product: Product, option: str, seed: int, val_fraction: float) -> str:
    if (product.id, option) in HELD_OUT:
        return "test"
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    return "val" if int(digest, 16) % 10_000 < val_fraction * 10_000 else "train"


def _demo(*steps: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(s) for s in steps]


# --------------------------------------------------------------------------- families


def _add_to_cart(product: Product, option: str, quantity: int) -> dict[str, Any]:
    number = _NUMBER_WORDS.get(quantity, str(quantity))
    steps = [
        {"action": "type", "target": "Search", "text": product.search, "submit": True},
        {"action": "click", "target": product.name, "role": "link"},
        {"action": "type", "target": product.option_name, "text": option},
    ]
    if quantity != 1:
        steps.append({"action": "type", "target": "Quantity", "text": str(quantity)})
    steps += [{"action": "click", "target": "Add to Cart"}, {"action": "stop"}]
    return {
        "id": f"add_{product.id}_{_slug(option)}_x{quantity}",
        "family": "add_to_cart",
        "intent": f"Add {number} {_phrase(product, option, quantity)} to the cart.",
        "success": [{"check": "cart_exact", "items": [_item(product, option, quantity)]}],
        "subgoals": [
            {"id": "searched", "check": "search_contains", "text": product.search},
            {"id": "on_product_page", "check": "page_is", "route": f"product/{product.id}"},
            {"id": "product_in_cart", "check": "cart_contains", "product_id": product.id},
        ],
        "demo": _demo(*steps),
    }


def _update_quantity(
    rng: random.Random, product: Product, option: str, start: int, target: int, distract: bool
) -> dict[str, Any]:
    label = _label(product, option)
    cart = [_item(product, option, start)]
    others = _distractors(rng, product, 1) if distract else []
    cart += [_item(p, o, q) for p, o, q in others]
    rng.shuffle(cart)
    intent = f"Change the quantity of {label} in the cart to {target}."
    if others:
        intent += " Leave the other items unchanged."
    expected = [dict(i, quantity=target) if i["product_id"] == product.id else i for i in cart]
    return {
        "id": f"update_{product.id}_{_slug(option)}_{start}_to_{target}",
        "family": "update_quantity",
        "intent": intent,
        "initial_state": {"cart": cart},
        "success": [{"check": "cart_exact", "items": expected}],
        "subgoals": [{"id": "on_cart_page", "check": "page_is", "route": "cart"}],
        "demo": _demo(
            {"action": "click", "target": "Cart", "role": "link"},
            {"action": "type", "target": f"Quantity for {label}", "text": str(target)},
            {"action": "click", "target": f"Update {label}"},
            {"action": "stop"},
        ),
    }


def _remove_item(
    rng: random.Random, product: Product, option: str, distractor_count: int
) -> dict[str, Any]:
    label = _label(product, option)
    others = _distractors(rng, product, distractor_count)
    cart = [_item(product, option, rng.choice((1, 2)))] + [_item(p, o, q) for p, o, q in others]
    rng.shuffle(cart)
    expected = [i for i in cart if i["product_id"] != product.id]
    return {
        "id": f"remove_{product.id}_{_slug(option)}_from_{len(cart)}",
        "family": "remove_item",
        "intent": f"Remove the {label} from the cart and keep everything else.",
        "initial_state": {"cart": cart},
        "success": [{"check": "cart_exact", "items": expected}],
        "subgoals": [{"id": "on_cart_page", "check": "page_is", "route": "cart"}],
        "demo": _demo(
            {"action": "click", "target": "Cart", "role": "link"},
            {"action": "click", "target": f"Remove {label}"},
            {"action": "stop"},
        ),
    }


# --------------------------------------------------------------------------- entry points


def generate_tasks(
    seed: int = DEFAULT_SEED, val_fraction: float = DEFAULT_VAL_FRACTION
) -> list[dict[str, Any]]:
    """Enumerate every task with its split assigned. Deterministic for a given seed."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    specs: list[tuple[dict[str, Any], Product, str]] = []
    for product, option in _variants():
        for quantity in ADD_QUANTITIES:
            specs.append((_add_to_cart(product, option, quantity), product, option))
    for product, option in _variants():
        for start, target, distract in UPDATE_CASES:
            rng = random.Random(f"{seed}:update:{product.id}:{option}:{start}:{target}")
            task = _update_quantity(rng, product, option, start, target, distract)
            specs.append((task, product, option))
    for product, option in _variants():
        for count in REMOVE_DISTRACTOR_COUNTS:
            rng = random.Random(f"{seed}:remove:{product.id}:{option}:{count}")
            specs.append((_remove_item(rng, product, option, count), product, option))

    tasks: list[dict[str, Any]] = []
    for task, product, option in specs:
        split = _split(task["id"], product, option, seed, val_fraction)
        # Same field order as the hand-written sample file: id, family, split, intent, ...
        ordered = {"id": task["id"], "family": task["family"], "split": split}
        ordered.update({k: v for k, v in task.items() if k not in ordered})
        Task.model_validate(ordered)  # fail fast on unknown checks or malformed demos
        tasks.append(ordered)

    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise RuntimeError("generated duplicate task ids")
    return tasks


def write_task_file(
    path: str | Path, seed: int = DEFAULT_SEED, val_fraction: float = DEFAULT_VAL_FRACTION
) -> list[dict[str, Any]]:
    tasks = generate_tasks(seed, val_fraction)
    payload = {
        "description": (
            "Generated mock-shop tasks across three workflow families with deterministic "
            "train/val/test splits. Regenerate with: python -m browser_agent.generate_tasks"
        ),
        "generator": {
            "module": "browser_agent.generate_tasks",
            "seed": seed,
            "val_fraction": val_fraction,
            "held_out_variants": sorted(f"{p}:{o}" for p, o in HELD_OUT),
        },
        "tasks": tasks,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate mock-shop tasks with splits.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Task file to write")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    args = parser.parse_args(argv)

    tasks = write_task_file(args.out, args.seed, args.val_fraction)
    counts: dict[str, dict[str, int]] = {}
    for t in tasks:
        counts.setdefault(t["family"], {}).setdefault(t["split"], 0)
        counts[t["family"]][t["split"]] += 1
    for family, splits in counts.items():
        row = "  ".join(f"{s}={splits.get(s, 0)}" for s in ("train", "val", "test"))
        print(f"{family:<18} {row}")
    totals = {s: sum(1 for t in tasks if t["split"] == s) for s in ("train", "val", "test")}
    print(f"{'total':<18} " + "  ".join(f"{s}={n}" for s, n in totals.items()))
    print(f"Wrote {len(tasks)} tasks to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
