<p align="center">
  <img src="logo.png" alt="PathoAI" width="440">
</p>

<p align="center">
  <b>An AI second reader for pathology quality assurance.</b><br>
  Assist, not diagnose.
</p>

<p align="center">
  <a href="https://pathoai.duckdns.org">🔬 Live app</a>
</p>

---

## What it is

PathoAI is a **second-reader QA tool** for histopathology. A pathologist (or an intern on their behalf) uploads an H&E image or pastes a report, and PathoAI gives an **independent read** and flags cases worth a second look. It never changes a sign-out and it does **not** diagnose — it is a safety net that surfaces disagreements for human re-review.

## Features

- **Analyze a case** — upload an H&E patch or a whole-slide image; get an independent read (tumour / normal / uncertain) with a confidence and a tumour heatmap.
- **Report check** — paste a pathology report; PathoAI classifies the conclusion into clinical categories and compares it against its own read.
- **Disagreement flag** — when the pathologist's conclusion and PathoAI's read differ, the case is flagged as worth a second look.
- **Review queue** — a worklist of the cases the model disagrees with, highest suspicion first.
- **Self-learning** — every submission is stored locally and folds into the next fine-tuning round, so the model sharpens on your lab's real cases over time. A live counter shows the growing database.
- **On-device** — images never leave the machine; every decision is appended to a local audit log.

## How it works

- **Model:** a ResNet18 classifier trained on open histopathology data — **PatchCamelyon** (~262k lymph-node patches) merged with **PathMNIST** (~90k colorectal patches), ~352k images total. Test AUROC ≈ 0.93.
- **Calibration:** temperature scaling for trustworthy probabilities.
- **Whole slides:** the image is tiled, each tile scored, and a top-K mean pooling produces a slide-level score with a heatmap.
- **Blended read:** the trained model is combined with a stain-density (hematoxylin) heuristic; when they disagree the case is surfaced for review.
- **Explainability:** Grad-CAM tile overlays show where the model is looking.

## Tech

- **Backend:** Python standard-library `http.server` (zero framework, on-prem / air-gap friendly) + PyTorch.
- **Frontend:** single-file React (loaded via CDN), guided step-by-step flow.
- **Model:** PyTorch / torchvision ResNet18, bundled as `model.pt`.

## Repo layout

- `backend/server.py` — HTTP server and routing (stdlib `ThreadingHTTPServer`, no framework).
- `backend/qa_service.py` — `QASession`: inference, feedback storage, fine-tuning, audit log.
- `pathqa/` — model definition, data loaders, slide tiling, Grad-CAM, calibration, CLI.
- `frontend/` — `index.html` + `app.jsx` (React via CDN; run `frontend/vendor_offline.sh` for air-gapped use).
- `model.pt` — trained ResNet18 weights (45 MB, committed). Same checkpoint the live site runs; its held-out metrics (test AUROC 0.929, accuracy 0.820, ECE 0.020) are what `/api/model-info` reports.
- `training/` — the scripts that trained `model.pt` and that fine-tune it (see Training).
- `DOCS.md` — longer design notes carried over from the original PathQA build: QA logic, calibration, workload budget, API, tuning `tau`.
- `setup_oracle.sh`, `setup_domain.sh` — VM deploy scripts (see Deploy).

## Run locally

```bash
git clone https://github.com/shubhambhandari126/pathoai.git && cd pathoai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=.:backend python3 backend/server.py --host 127.0.0.1 --port 8000
# open http://localhost:8000
```

`./run.sh` does the same but binds to all interfaces and pre-loads a 24-case / 3-slide demo worklist.
The server picks up `model.pt` from the repo root automatically. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PATHQA_MODEL` | `model.pt` | model weights to load; fine-tuning also saves here |
| `PATHQA_FEEDBACK_DIR` | `data/feedback` | where submitted cases are stored for self-learning |
| `PATHQA_AUDIT_LOG` | `audit_log.jsonl` | append-only audit log |
| `PATHQA_PASSWORD` | unset | set it to require an access code before the UI loads |
| `PORT` | `8000` | overrides `--port` |

`data/` and `audit_log.jsonl` are git-ignored so lab data is never committed.

## Training

`model.pt` was produced with the scripts in `training/` on a GPU (Google Colab works) and then committed here; the server loads it as-is. The committed checkpoint's embedded note says it was "trained/continued on PatchCamelyon (HF mirror)", i.e. `train_pcam_hf.py --base` continuing from a PathMNIST model made with `train_open.py`, which is the PathMNIST + PatchCamelyon merge described above. Run the scripts from the repo root so `pathqa` is importable:

```bash
pip install medmnist datasets   # the open datasets the trainers download
PYTHONPATH=. python3 training/train_open.py --epochs 3 --out model.pt
PYTHONPATH=. python3 training/train_pcam_hf.py --base model.pt --epochs 2 --out model.pt
```

| Script | Purpose |
|---|---|
| `train_open.py` | PathMNIST (colorectal, ~100k tiles, auto-download), binarised tumour vs rest |
| `train_pcam_hf.py` | PatchCamelyon (~327k lymph-node tiles) via the Hugging Face `datasets` mirror; `--base` continues from an existing checkpoint |
| `train_combined.py` | both datasets mixed into one run (avoids forgetting the first) |
| `train_pcam.py` | PatchCamelyon via torchvision (7 GB download); `--synthetic` checks the pipeline with no download |
| `train_on_your_data.py` | fine-tune on a lab's own `train/{normal,tumor}` and `val/{normal,tumor}` folders |
| `finetune_from_feedback.py` | fine-tune on cases submitted through the app (`data/feedback/`) |

Every script saves the weights plus the fitted temperature and held-out metrics in one checkpoint, which is what `/api/model-info` reports. See `DOCS.md` for the design notes behind the QA logic and calibration.

## Deploy

- `setup_oracle.sh <access-code>` — one-shot installer for an Ubuntu VM (adds swap on small boxes, installs deps, clones this repo, installs its committed `model.pt`, sets up a systemd service, opens the firewall). The live app runs this way on an Oracle Always-Free VM. On the VM: `curl -sL https://raw.githubusercontent.com/shubhambhandari126/pathoai/main/setup_oracle.sh | sudo bash -s -- <access-code>`. Re-running it on an existing VM re-clones `main`, refreshes `model.pt` and restarts the service.
- `setup_domain.sh <domain>` — puts the app behind a domain with automatic free HTTPS via Caddy + Let's Encrypt (moves the app to port 8080 and proxies 443 to it).

## Disclaimer

PathoAI is a **quality-assurance aid, not a certified diagnostic device**. It outputs re-check recommendations only. Any clinical use requires proper validation, governance, and human oversight. Do not use it to make a diagnosis.
