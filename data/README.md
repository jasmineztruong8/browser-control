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
| `mock_shop.json` | 8 sample tasks for the local mock shop, across three families (add to cart, update quantity, remove item) and all three splits |

## Demonstrations

Generate verified demonstrations by running the scripted policy and keeping only successful episodes:

```bash
python -m browser_agent.rollout --set policy.name=scripted tasks.split=train run.name=demos
```

The trajectories are written to `runs/<run_id>/trajectories.jsonl`.
