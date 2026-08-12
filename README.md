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

## Run locally

```bash
unzip pathoai_app.zip && cd pathoai
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision pillow numpy h5py
PYTHONPATH=.:backend python3 backend/server.py --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Deploy

- `setup_oracle.sh` — one-shot installer for an Ubuntu VM (installs deps, pulls the app, sets up a systemd service, opens the firewall). The live app runs this way on an Oracle Always-Free VM.
- `setup_domain.sh` — puts the app behind a domain with automatic free HTTPS via Caddy + Let's Encrypt.

## Disclaimer

PathoAI is a **quality-assurance aid, not a certified diagnostic device**. It outputs re-check recommendations only. Any clinical use requires proper validation, governance, and human oversight. Do not use it to make a diagnosis.
