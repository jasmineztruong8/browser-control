# Demo

The runnable browser-agent demo will live here and reuse the shared code in `src/browser_agent/`.

Until then, you can watch the agent work through a task in a visible browser window:

```bash
python -m browser_agent.rollout --set env.headless=false env.settle_ms=800 tasks.limit=1
```
