#!/usr/bin/env bash
# Runs ON the VM (scripts/gcp/vm.sh setup calls it). Installs what the pilot, SFT and RL need:
# Python 3.11 venv with the train extras, Chromium for Playwright, and a Jupyter kernel.
# Safe to rerun, e.g. after dependencies change.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== GPU driver (the first boot installs it, which takes a few minutes)"
for _ in $(seq 60); do
  nvidia-smi >/dev/null 2>&1 && break
  sleep 10
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv ||
  { echo "error: no NVIDIA driver after 10 minutes; try: sudo reboot" >&2; exit 1; }

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq tmux >/dev/null

echo "== Python environment"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -q -e ".[dev,train]" papermill ipykernel
# A named kernel, so notebooks run with this venv rather than the image's own Python.
.venv/bin/python -m ipykernel install --user --name browser-agent --display-name "browser-agent" >/dev/null

echo "== Chromium for Playwright"
sudo .venv/bin/playwright install-deps chromium >/dev/null
.venv/bin/playwright install chromium

[[ -f .env ]] || cp .env.example .env

echo "== Check"
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch can't see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.0f} GB, "
      f"bf16={torch.cuda.is_bf16_supported()}")
EOF
.venv/bin/python -m browser_agent.rollout --set tasks.limit=1 | tail -1
echo "Setup done. Next, from your machine: scripts/gcp/vm.sh pilot"
