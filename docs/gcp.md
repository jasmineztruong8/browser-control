# Running on Google Cloud (GCP)

How to get a GPU VM on Google Cloud and run the compute pilot on it. The same VM, and the same setup, works for SFT and RL training. You run everything from your own terminal through `scripts/gcp/vm.sh`; you never have to click through the GCP console except for the one-time steps below.

## What you get

| | Default |
| --- | --- |
| Machine | `g2-standard-8`: 8 vCPUs, 32 GB RAM, 1× NVIDIA L4 (24 GB, supports bf16) |
| Image | Google's Deep Learning VM (Ubuntu with CUDA; the NVIDIA driver installs on first boot) |
| Disk | 150 GB |
| Safety net | the VM stops itself 8 hours after each start |

**Cost.** A `g2-standard-8` costs roughly $0.70–0.90 per hour while it's running in `us-central1`. Check the [GPU pricing page](https://cloud.google.com/compute/gpus-pricing) for current numbers. A stopped VM costs only its disk, about $15 a month for 150 GB. A pilot run is well under an hour. To save more, set `SPOT=true` for a Spot VM at roughly a third of the price. Google can stop a Spot VM at any time, which is fine for the pilot but not for long training runs without checkpoints.

All settings are in [`scripts/gcp/config.env`](../scripts/gcp/config.env) with comments. For example, `MACHINE_TYPE=n1-standard-8 GPU_TYPE=nvidia-tesla-t4` gives a cheaper T4.

## One-time setup

### 1. Install and sign in to the Google Cloud CLI

Install it from <https://cloud.google.com/sdk/docs/install> (on a Mac: `brew install --cask google-cloud-sdk`). Then sign in:

```bash
gcloud auth login
```

### 2. Pick the project

All GCP resources live in a *project*, and costs go to that project's *billing account*. The team should share **one** project, so everyone can use the same VM and the course credits land in one place.

- **If a teammate already set one up**, ask them for the project ID and to add you (step 4). Then skip to step 5.
- **To create one:**
  1. In the [console](https://console.cloud.google.com/projectcreate), create a project, e.g. `browser-agent-coms6998`. Note its **project ID**, which may differ slightly from the name.
  2. Under *Billing*, link it to the billing account that holds your course credits. Credit coupons usually create a billing account when you redeem them.
  3. Turn on Compute Engine:

     ```bash
     gcloud services enable compute.googleapis.com --project YOUR_PROJECT_ID
     ```

### 3. Request GPU quota

New projects usually start with a GPU quota of **0**, so the VM can't be created until you request more. Approval often takes minutes but can take a day or two, so do this early.

1. Open [*IAM & Admin → Quotas & System Limits*](https://console.cloud.google.com/iam-admin/quotas) for the project.
2. Search for **GPUs (all regions)** and request at least **1**.
3. Search for **NVIDIA L4 GPUs**, pick the region `us-central1`, and request **1**. For a Spot VM, request **Preemptible NVIDIA L4 GPUs** instead. For a T4, use the T4 quotas.

### 4. Give teammates access

The project owner runs this once for each teammate:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member=user:TEAMMATE_EMAIL --role=roles/editor
```

`roles/editor` is the simplest choice for a course project. For narrower access, grant `roles/compute.instanceAdmin.v1` and `roles/iam.serviceAccountUser` instead.

### 5. Tell the scripts which project to use

Create `scripts/gcp/config.local.env`. Git ignores this file, so each person keeps their own:

```bash
PROJECT=YOUR_PROJECT_ID
```

The scripts never fall back to gcloud's default project, so a VM can't end up in an unrelated project by accident.

## Run the compute pilot

```bash
scripts/gcp/vm.sh create   # about 1 minute
scripts/gcp/vm.sh setup    # about 10 minutes the first time: GPU driver, Python packages, Chromium
scripts/gcp/vm.sh pilot    # starts the pilot in the background on the VM
scripts/gcp/vm.sh logs     # follow it; Ctrl-C stops following, not the pilot
scripts/gcp/vm.sh fetch    # when it's done: copy results to runs/gcp/ here
scripts/gcp/vm.sh stop     # stop paying for the GPU
```

- **Branch.** The VM checks out the branch you're on locally, so push your branch first. Set `BRANCH=main` to override.
- **First SSH.** The first `gcloud compute ssh` creates an SSH key for you. Press Enter at the passphrase prompts.
- **Results.** `pilot` runs [`notebooks/compute_pilot.ipynb`](../notebooks/compute_pilot.ipynb) on the VM without opening it. `fetch` copies back the summary table (`runs/gcp/<timestamp>-pilot/summary.md`) and the executed notebook with every cell's output.
- **Overrides.** Config changes go after `pilot` in the same `KEY=VALUE` form as `--set`:

  ```bash
  scripts/gcp/vm.sh pilot policy.load_in_4bit=false          # full-precision weights
  scripts/gcp/vm.sh pilot policy.model_name=Qwen/Qwen2.5-1.5B-Instruct
  ```

- **Disconnects.** The pilot runs in `tmux` on the VM, so closing your laptop or losing Wi-Fi doesn't stop it.

## Day-to-day

| Command | What it does |
| --- | --- |
| `scripts/gcp/vm.sh status` | Is the VM running? |
| `scripts/gcp/vm.sh start` | Start a stopped VM. Its disk, and everything installed on it, is kept. |
| `scripts/gcp/vm.sh ssh` | Open a shell on the VM. The repo is in `~/browser-control`, with its environment in `.venv`. |
| `scripts/gcp/vm.sh setup` | Rerun after dependencies change. It's safe to repeat. |
| `scripts/gcp/vm.sh stop` | Stop the VM. **Do this whenever you're done.** |
| `scripts/gcp/vm.sh delete` | Remove the VM and its disk for good. |

On the VM, the rest of the repo works as usual, e.g. `.venv/bin/python -m browser_agent.rollout`.

## Troubleshooting

- **`ZONE_RESOURCE_POOL_EXHAUSTED` or "does not have enough resources".** GPUs are in short supply in that zone right now. Try another: `ZONE=us-central1-b scripts/gcp/vm.sh create`. Other zones with L4s include `us-central1-c`, `us-east4-a` and `us-west1-a`. Your quota must cover the zone's region.
- **`Quota 'GPUS_ALL_REGIONS' exceeded`** or a similar quota error. Step 3 hasn't been approved yet, or it was approved for a different region.
- **`setup` says there's no NVIDIA driver.** The first boot can take a while. Wait a minute and run `setup` again. If that fails, run `scripts/gcp/vm.sh ssh`, then `sudo reboot`.
- **`There was a problem refreshing your current auth tokens`.** Your sign-in expired. Run `gcloud auth login` again.
- **The VM stopped on its own.** That's the `MAX_RUN_HOURS` safety net, or a Spot VM being reclaimed. Run `scripts/gcp/vm.sh start`. For long training runs, raise the limit: `MAX_RUN_HOURS=24 scripts/gcp/vm.sh create`. For an existing VM, change it in the console.
