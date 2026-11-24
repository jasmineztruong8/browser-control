# Data

## Tasks (`data/tasks/`)

Each task file is JSON: `{"tasks": [ ... ]}`. A task looks like this:

```json
{
  "id": "add_mug_blue_1",
  "family": "add_to_cart",
  "split": "train",
  "intent": "Add one blue Ceramic Coffee Mug to the cart.",
  "initial_state": {"cart": []},
  "success": [{"check": "cart_exact", "items": [{"product_id": "mug", "option": "Blue", "quantity": 1}]}],
  "subgoals": [{"id": "searched", "check": "search_contains", "text": "mug"}],
  "demo": [{"action": "type", "target": "Search", "text": "mug", "submit": true}, {"action": "stop"}]
}
```

| Field | Meaning |
| --- | --- |
| `family` | Workflow family, used for filtering (`tasks.families`) and per-family results |
| `split` | `train`, `val` or `test`. Test tasks must never be used for demonstrations or tuning |
| `initial_state` | State the environment applies on reset (e.g. a pre-filled cart) |
| `success` | Checks that must all hold on the final state |
| `subgoals` | Checks for intermediate progress, used by the `subgoal` reward. Each needs a unique `id` |
| `start_path` | Optional suffix added to the start URL (e.g. `#cart`) |
| `demo` | Optional scripted solution. `target` is matched against element labels on the live page |

The available checks are documented in `src/browser_agent/checks.py`. Loading a task file fails if it uses an unknown check, repeats a task id or leaves out a subgoal id.

### Files

| File | Contents |
| --- | --- |
| `mock_shop.json` | 8 hand-written sample tasks for the local mock shop, across three families (add to cart, update quantity, remove item) and all three splits. Used by the tests and the README quickstart |
| `mock_shop_v1.json` | 108 generated tasks for the mock shop with versioned splits (65 train / 16 val / 27 test). Use this file for demonstrations, training and evaluation. See [Generating tasks](#generating-tasks) |

## Generating tasks

`mock_shop_v1.json` is produced by `src/browser_agent/generate_tasks.py`, which enumerates the mock shop catalog (5 products, 12 product/option variants) across the three families:

| Family | Instances | How they vary |
| --- | --- | --- |
| `add_to_cart` | 48 | every variant x quantity in {1, 2, 3, 5} |
| `update_quantity` | 36 | every variant x three (start, target) quantity pairs; two of the three also have another item in the cart that must be left alone |
| `remove_item` | 24 | every variant with one or two other items in the cart that must be kept |

Splits are deterministic and versioned in the file:

- **Test** holds out three product variants (`mug:Green`, `tshirt:L`, `bottle:500 ml`). They are only ever the *target* of a test task and never the target of a train or val task, so the test split measures generalization to variants the model was never shown. Each product still has at least two options in train, so the model must learn to choose, not memorize.
- **Train / val** are assigned by hashing the task id with the seed (val fraction 0.15). Changing the seed moves tasks between train and val but never changes the test set.

Do not edit `mock_shop_v1.json` by hand; a test checks that it matches the generator. To change the task set, edit the generator and regenerate (bump the file version if the change alters existing tasks):

```bash
python -m browser_agent.generate_tasks --out data/tasks/mock_shop_v1.json
```

A browser test (`tests/test_generate_tasks.py`) runs every generated demo and checks it succeeds, so every task in the file is a verified demonstration.

## Demonstrations

Generate verified demonstrations by running the scripted policy and keeping only successful episodes:

```bash
python -m browser_agent.rollout --set tasks.path=data/tasks/mock_shop_v1.json tasks.split=train run.name=demos
```

The trajectories are written to `runs/<run_id>/trajectories.jsonl`. Every task in `mock_shop_v1.json` has a working demo, so the run should report `success_rate: 1.0`; drop any episode that did not succeed before using the file for SFT.
