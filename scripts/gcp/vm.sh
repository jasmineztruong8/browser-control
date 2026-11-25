#!/usr/bin/env bash
# Manage the team's GPU VM on Google Cloud. Run from your own machine; see docs/gcp.md.
#
#   scripts/gcp/vm.sh create            create the VM (settings in scripts/gcp/config.env)
#   scripts/gcp/vm.sh setup             clone the repo on the VM and install everything
#   scripts/gcp/vm.sh pilot [KEY=VALUE] run the compute pilot in the background on the VM
#   scripts/gcp/vm.sh logs              follow the pilot's output (Ctrl-C stops following only)
#   scripts/gcp/vm.sh fetch             copy the VM's runs/ to runs/gcp/ here
#   scripts/gcp/vm.sh ssh               open a shell on the VM
#   scripts/gcp/vm.sh status | start | stop | delete
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
REMOTE_DIR="browser-control" # relative to the VM user's home directory

# Settings from the command-line environment win, then config.local.env, then config.env.
# config.env only fills in variables that are still unset, so loading local first is enough.
# shellcheck source=/dev/null
[[ -f "$HERE/config.local.env" ]] && source "$HERE/config.local.env"
# shellcheck source-path=SCRIPTDIR
source "$HERE/config.env"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -n "$PROJECT" ]] || die "set PROJECT in scripts/gcp/config.local.env (see docs/gcp.md)"
command -v gcloud >/dev/null || die "gcloud not found: install the Google Cloud CLI first"

GC=(gcloud --project "$PROJECT")
ZONE_ARGS=(--zone "$ZONE")

remote() { "${GC[@]}" compute ssh "$VM_NAME" "${ZONE_ARGS[@]}" --command "$1"; }

current_branch() {
  if [[ -n "$BRANCH" ]]; then
    echo "$BRANCH"
    return
  fi
  local b
  b="$(git -C "$REPO_ROOT" branch --show-current)"
  if ! git -C "$REPO_ROOT" ls-remote --exit-code --heads origin "$b" >/dev/null 2>&1; then
    die "branch '$b' isn't on GitHub yet; push it or set BRANCH=main"
  fi
  if [[ -n "$(git -C "$REPO_ROOT" log "origin/$b..$b" --oneline 2>/dev/null)" ]]; then
    echo "warning: '$b' has local commits that aren't pushed; the VM won't see them" >&2
  fi
  echo "$b"
}

# Clone or update the repo on the VM at the chosen branch.
sync_code() {
  local branch
  branch="$(current_branch)"
  echo "VM will use branch: $branch"
  remote "set -e
    [ -d $REMOTE_DIR/.git ] || git clone -q $REPO_URL $REMOTE_DIR
    cd $REMOTE_DIR
    git fetch -q origin
    git checkout -q $branch
    git pull -q --ff-only origin $branch
    git log -1 --format='VM code: %h %s'"
}

newest_image_family() {
  # Filtered here rather than with --filter, which rejects regex combinations like these.
  gcloud compute images list --project deeplearning-platform-release --no-standard-images \
    --format="value(family)" | grep -E '^common-cu.*ubuntu' | grep -v -E 'arm|aarch' |
    sort -uV | tail -1
}

cmd_create() {
  local family="$IMAGE_FAMILY"
  if [[ -z "$family" ]]; then
    family="$(newest_image_family)"
    [[ -n "$family" ]] || die "couldn't find a Deep Learning VM image; set IMAGE_FAMILY"
  fi
  local args=(
    --machine-type "$MACHINE_TYPE"
    --image-project deeplearning-platform-release
    --image-family "$family"
    --boot-disk-size "${DISK_GB}GB"
    --boot-disk-type pd-balanced
    --maintenance-policy TERMINATE # required for VMs with GPUs
    --metadata install-nvidia-driver=True
    --scopes cloud-platform
  )
  [[ -n "$GPU_TYPE" ]] && args+=(--accelerator "type=$GPU_TYPE,count=$GPU_COUNT")
  [[ "$SPOT" == "true" ]] && args+=(--provisioning-model SPOT)
  [[ "$MAX_RUN_HOURS" != "0" ]] && args+=(--max-run-duration "${MAX_RUN_HOURS}h")
  if [[ "$SPOT" == "true" || "$MAX_RUN_HOURS" != "0" ]]; then
    args+=(--instance-termination-action STOP)
  fi

  echo "Creating $VM_NAME ($MACHINE_TYPE${GPU_TYPE:+ + $GPU_TYPE}, image $family) in $PROJECT/$ZONE"
  "${GC[@]}" compute instances create "$VM_NAME" "${ZONE_ARGS[@]}" "${args[@]}"
  echo
  echo "Created. Next: scripts/gcp/vm.sh setup"
  [[ "$MAX_RUN_HOURS" != "0" ]] && echo "The VM stops itself ${MAX_RUN_HOURS}h after each start."
  return 0
}

cmd_setup() {
  sync_code
  remote "bash $REMOTE_DIR/scripts/gcp/remote_setup.sh"
}

cmd_pilot() {
  sync_code
  # Pass overrides through the environment; the notebook appends PILOT_OVERRIDES to OVERRIDES.
  local overrides="${*:-}"
  remote "set -e
    cd $REMOTE_DIR
    if tmux has-session -t pilot 2>/dev/null; then
      echo 'A pilot is already running. Follow it with: scripts/gcp/vm.sh logs'; exit 1
    fi
    mkdir -p runs
    tmux new-session -d -s pilot \"PILOT_OVERRIDES='$overrides' bash scripts/gcp/remote_pilot.sh 2>&1 | tee runs/pilot.log\"
    echo 'Pilot started on the VM. It keeps running if you disconnect.'"
  echo "Follow it with: scripts/gcp/vm.sh logs     When it's done: scripts/gcp/vm.sh fetch"
}

cmd_logs() { remote "tail -n 200 -f $REMOTE_DIR/runs/pilot.log"; }

cmd_fetch() {
  local dest="$REPO_ROOT/runs/gcp"
  mkdir -p "$dest"
  "${GC[@]}" compute scp --recurse "${ZONE_ARGS[@]}" "$VM_NAME:$REMOTE_DIR/runs/*" "$dest/"
  echo "Copied to $dest"
}

case "${1:-}" in
  create) cmd_create ;;
  setup) cmd_setup ;;
  pilot) shift && cmd_pilot "$@" ;;
  logs) cmd_logs ;;
  fetch) cmd_fetch ;;
  ssh) "${GC[@]}" compute ssh "$VM_NAME" "${ZONE_ARGS[@]}" ;;
  status)
    "${GC[@]}" compute instances describe "$VM_NAME" "${ZONE_ARGS[@]}" \
      --format="table(name,status,machineType.basename(),scheduling.provisioningModel)"
    ;;
  start) "${GC[@]}" compute instances start "$VM_NAME" "${ZONE_ARGS[@]}" ;;
  stop) "${GC[@]}" compute instances stop "$VM_NAME" "${ZONE_ARGS[@]}" ;;
  delete) "${GC[@]}" compute instances delete "$VM_NAME" "${ZONE_ARGS[@]}" ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
