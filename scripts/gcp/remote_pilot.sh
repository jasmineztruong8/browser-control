#!/usr/bin/env bash
# Runs ON the VM (scripts/gcp/vm.sh pilot starts it in tmux). Executes the compute pilot
# notebook without a browser UI and saves the executed copy, with all its output, in runs/.
# Extra config overrides come in through PILOT_OVERRIDES, which the notebook reads.
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p runs
executed="runs/$(date +%Y%m%d-%H%M%S)-compute_pilot.ipynb"
echo "Overrides: ${PILOT_OVERRIDES:-(none)}"
.venv/bin/papermill notebooks/compute_pilot.ipynb "$executed" \
  --kernel browser-agent --cwd notebooks --log-output --no-progress-bar

echo
echo "== Summary"
latest="$(printf '%s\n' runs/*-pilot/ | sort | tail -1)" # run dirs start with a timestamp
cat "${latest}summary.md"
echo
echo "Executed notebook: $executed"
