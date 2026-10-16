> **About this file.** These are the original PathQA / Pathify design notes, carried over
> into this repo from the Hugging Face Space `Shubhambhandari/Pathify` in October 2026.
> The app is now called **PathoAI** and lives at https://github.com/shubhambhandari126/pathoai;
> see [README.md](README.md) for how to run, train and deploy it. Paths below have been
> updated to this repo's layout (the training scripts are under `training/`). The
> `demo.py`, `slide_demo.py` and `pathqa-app` launcher scripts mentioned below were not
> carried over; use `backend/server.py` / `run.sh` and the `training/` scripts instead.

# PathQA — an AI second-reader QA safety-net for pathology labs

**Positioning:** quality assurance / workflow, **not** diagnosis. PathQA never
issues a diagnosis and never changes a pathologist's sign-out. It runs quietly
behind the sign-out and does exactly one thing: when it *confidently disagrees*
with the pathologist, it raises a hand — "case N, worth a second look, here's
why." A human decides. The pathologist stays in control.

This repo is the **v1 build**, in two stages exactly as scoped:

1. **Patch-classifier demo** on public data (PatchCamelyon) — proves the model +
   the QA logic end-to-end.
2. **Train-on-your-data** version — the model comes to the lab's own labeled
   slides, which is what fixes domain shift, and nothing leaves the machine.

---

## Why this design (the three product requirements)

**1. Trains on the lab's own labeled slides → solves domain shift + keeps data private.**
Off-the-shelf histopathology models degrade badly across stains, scanners and
tissue mixes. `training/train_on_your_data.py` fine-tunes on a folder of the lab's own
patches, so the model learns *that* lab's distribution. All I/O is local — the
model comes to the data.

**2. Flags only disagreements → a human re-checks.**
The QA engine (`pathqa/qa.py`) compares the model's read to the pathologist's
label and flags a case only when the model **disagrees** *and* is **confident**
(`confidence ≥ tau`). Everything else passes untouched. Output is a prioritized
re-check worklist, sorted by confidence — the queue a pathologist actually opens.

**3. Assist, not replace.**
No case is ever auto-relabeled. The only output is a "recommend re-review" flag
with a plain-language reason. Every artifact carries that reminder.

Two extra design choices that matter for a *QA* tool specifically:

- **Calibration.** A QA threshold is only meaningful if "0.9 confidence" behaves
  like 0.9. Raw softmax is over-confident, so I apply **temperature scaling**
  (Guo et al., 2017) on the validation split. In the smoke test this dropped
  validation loss from 0.265 → 0.231.
- **Workload budget.** Instead of a fixed `tau`, a lab can say "flag the top 5%
  most-confident disagreements" (`--budget 0.05`) so the QA queue fits the time
  the lab actually has. The engine backs out the effective `tau` for you.

---

## Layout

```
pathqa/
  pathqa-app / .bat       # ONE-COMMAND LAUNCHER (venv + deps + open browser)
  pathqa/
    model.py     # ResNet18/34 binary tumor/normal head + TemperatureScaler
    data.py      # 3 sources: PatchCamelyon | your ImageFolder | synthetic (tests)
    engine.py    # train / evaluate (AUROC, ECE) / temperature fitting
    qa.py        # SecondReader — patch-level disagreement + flagging engine
    wsi.py       # whole-slide tiling + tissue detection
    slide.py     # slide aggregation (top-K), slide-level QA, heatmap
    gradcam.py   # Grad-CAM explainer (why the model flagged it)
    cli.py       # `pathqa serve|demo|slide-demo|train` dispatcher
  backend/       # stdlib HTTP server + stateful QA session
  frontend/      # single-file React UI (Patch QA + Whole-slide QA tabs)
  demo.py                 # patch QA demo (stage 1)
  slide_demo.py           # whole-slide QA demo
  training/train_on_your_data.py   # fine-tune on the lab's own labeled patches (stage 2)
  training/train_pcam.py           # real PatchCamelyon training + AUROC/ECE
  requirements.txt
```

## Install

```bash
pip install -r requirements.txt   # torch, torchvision, h5py
```

## Run — Stage 1: patch-classifier demo

No download, verifies the whole pipeline in ~1 minute on CPU:

```bash
python demo.py --synthetic --epochs 1 --no-pretrained
```

Real public data (downloads PatchCamelyon on first run; use a GPU or Colab):

```bash
python demo.py --pcam --subset 20000 --epochs 3 --budget 0.05
```

What it does: loads patches → fine-tunes ResNet18 → temperature-calibrates →
simulates a sign-out queue with a small injected human-error rate → runs the QA
second reader → prints a report and writes `qa_out/worklist.csv` +
`qa_out/qa_report.json`.

Sample output:

```
PathQA second-reader report
  cases reviewed        : 128
  model/pathologist disagreements: 17 (13.3%)
  flagged for re-check  : 6 (4.7%) at confidence >= 0.85
  -> 122 cases pass QA untouched; pathologist reviews only the 6 flagged.
[QA effectiveness] of 17 simulated sign-out errors, the second reader flagged 6 for re-check.
```

## Run — Stage 2: train on your data

Lay your labeled patches out as:

```
my_lab_data/
  train/normal/*.png   train/tumor/*.png
  val/normal/*.png     val/tumor/*.png
```

```bash
python training/train_on_your_data.py --data my_lab_data --epochs 5 --out model.pt
```

Saves the fine-tuned weights **and** the fitted temperature, so the QA engine
gets calibrated confidences out of the box. Everything stays on the machine.

---

## Run — the local app (one command)

A local, on-prem app so a pathologist (non-coder) can use it. The backend is
**Python stdlib only** (no framework to install, nothing phones home — chosen
for an air-gapped lab), and the frontend is a single-file React UI it serves.

```bash
./pathqa-app            # macOS/Linux — makes a venv, installs deps, opens the browser
pathqa-app.bat          # Windows

./pathqa-app demo       # patch QA demo (headless)
./pathqa-app slide-demo # whole-slide QA demo (headless)
./pathqa-app train --help
```

The launcher creates a `.venv`, installs `requirements.txt`, then starts the
server and opens `http://127.0.0.1:8000` (localhost only). Under the hood it
calls `pathqa/cli.py`; you can also run the pieces directly:

```bash
PYTHONPATH=. python backend/server.py --open
```

Out of the box it fine-tunes a small demo model on synthetic patches so there's
a working worklist immediately. To serve a model you trained on your own data:

```bash
PATHQA_MODEL=model.pt PYTHONPATH=. python backend/server.py
```

What the UI does: shows the flagged re-check queue (patch thumbnail, sign-out vs
model read, confidence), a live τ slider to tune the queue length, and two
actions per case — **Confirm sign-out** or **Amend (model was right)**. Every
decision is appended to `audit_log.jsonl` with model read, confidence and a note
— the governance trail. No sign-out is ever changed automatically.

API (stable; maps 1:1 onto FastAPI path operations if you later swap frameworks):
`GET /api/summary`, `GET /api/worklist`, `GET /api/patch/{id}.png`,
`POST /api/decision`, `POST /api/settings`.

Air-gapped frontend: run `frontend/vendor_offline.sh` once to pull React+Babel
locally, then point the three `<script>` tags in `index.html` at `./vendor/...`.

## Whole-slide QA (patch → slide)

A real sign-out is per-slide, not per-patch. The whole-slide layer tiles a slide,
scores every tissue tile, and aggregates into one slide-level decision:

- `pathqa/wsi.py` — tiling + tissue/background detection (keeps stained tissue,
  drops glass). Opens large PIL images, true WSI via OpenSlide (`.svs/.ndpi`,
  optional lazy import), or a synthetic slide for the demo.
- `pathqa/slide.py` — **top-K mean pooling**: a slide is called tumor if its most
  suspicious regions are tumor (matches metastasis detection — one focus is
  enough — far better than averaging tumor away across a big slide). Also emits a
  tumor-probability **heatmap** and the slide-level QA flags.

Run the slide demo (synthetic cohort, writes a heatmap per slide + a worklist):

```bash
python slide_demo.py --n-slides 8 --tau 0.7 --outdir slide_out
```

For a real slide, swap `SyntheticSlide` for `open_slide_image("case.svs")` — the
rest is unchanged. OpenSlide is only needed for native WSI formats:
`pip install openslide-python` (plus the system OpenSlide library).

In the app, the **Whole-slide QA** tab shows each flagged slide as a heatmap card
(sign-out vs model, slide score, suspicious-tile count) with Confirm / Amend, and
its own τ slider. Slide decisions log to the same `audit_log.jsonl` with
`"level":"slide"`.

## Analyze an image (real model + stain), and the train-better loop

The app's "Analyze image" tab runs an uploaded H&E image through BOTH methods
and combines them:

- the trained CNN (`pathqa/infer.py`) with test-time augmentation, giving a
  calibrated tumour probability plus an uncertainty estimate (so it can say
  "not sure" instead of guessing), and a Grad-CAM overlay (patch) or a tiled
  tumour heatmap with the top suspicious tiles (slide);
- the stain/colour method (dense-nuclei density) as an independent check.

The two are merged into one verdict: if they agree, confidence goes up; if they
disagree, it returns "needs review" rather than a false certainty.

Endpoints: `POST /api/analyze-image` (base64 image in, verdict plus two overlay
images out), `POST /api/feedback` (save a labelled image as training data),
`GET /api/feedback/stats`.

Train-better loop: every image you label in the app is saved under
`data/feedback/{normal,tumor}/`. When you have enough, fine-tune on exactly the
cases your lab sees:

```bash
python training/finetune_from_feedback.py --base pcam_model.pt --out pcam_model.pt
# prints held-out AUROC + ECE, saves a calibrated checkpoint the app loads
PATHQA_MODEL=pcam_model.pt ./pathqa-app
```

So it gets better the more it is used, and always reports honest metrics before
you adopt a new checkpoint.

## Explainability — Grad-CAM

A flag is only actionable if the pathologist can see *why* the model disagreed.
`pathqa/gradcam.py` runs Grad-CAM on the classifier's last conv block and
overlays a heat map on the patch showing which pixels drove the call. In the
app, clicking a flagged patch opens the patch next to its "model attention"
overlay. This is explanation, not diagnosis — it never changes a call.

## Real training on PatchCamelyon

`training/train_pcam.py` trains on real PCam and reports honest held-out metrics — AUROC
and Expected Calibration Error (ECE) before/after temperature scaling — then
saves an app-ready checkpoint.

```bash
# GPU (full run)
python training/train_pcam.py --epochs 5 --batch-size 256 --out pcam_model.pt

# quick CPU subset
python training/train_pcam.py --subset 8000 --epochs 1 --batch-size 64 --out pcam_model.pt

# pipeline check, no download
python training/train_pcam.py --synthetic --no-pretrained --epochs 1 --out demo_model.pt
```

Colab (free GPU) — PCam is ~7 GB so run it there or on a lab GPU box:

```python
!pip -q install torch torchvision h5py
!python training/train_pcam.py --epochs 5 --batch-size 256 --out pcam_model.pt
```

It prints something like `AUROC: 0.96 · ECE: 0.08 → 0.02 (calibrated)`. Then:

```bash
PATHQA_MODEL=pcam_model.pt ./pathqa-app     # the app now uses your real model
```

Note: full PCam training needs a GPU and the ~7 GB download; the synthetic
`--synthetic` path exists to verify the training/metrics/calibration code runs
end-to-end without either. The pipeline was verified this way (AUROC/ECE/
checkpoint all produced correctly); real clinical numbers come from your run.

## Key parameters

| Flag | Meaning |
|---|---|
| `--tau 0.9` | Flag disagreements at/above this confidence. |
| `--budget 0.05` | Instead of `tau`, flag only the top 5% most-confident disagreements (fits a workload). |
| `--path-error-rate` | (demo only) simulated pathologist error rate, to show the safety-net catching misses. |
| `--subset` / `--syn-train` | Cap dataset size for CPU-friendly runs. |

`tau` is the single most important dial. Lower it → catch more true misses but a
longer re-check queue; raise it → shorter queue but you may miss borderline
errors. Tune it against the lab's tolerated false-flag rate on a labeled
holdout, then re-check monthly.

---

## What was verified

Ran end-to-end on CPU (ARM64): training converges, temperature calibration
lowers validation loss, both threshold and workload-budget flagging modes
produce a correct prioritized worklist, and the JSON report round-trips. The
real-data path uses the same code with `--pcam` or `--data`.

## Roadmap (v2+, out of scope for this build)

- **Slide-level, not just patch-level:** tile a whole-slide image, aggregate
  patch scores into a slide decision, and flag at the slide level.
- **Uncertainty on top of calibration:** MC-dropout / deep ensembles to separate
  "confident disagreement" from "model has no idea."
- **Active learning loop:** feed re-checked flags back as new labels so the model
  improves on the lab's hard cases over time.
- **Human factors:** a viewer that jumps to the flagged region with a heatmap
  (Grad-CAM) so re-review takes seconds.
- **Governance:** audit log of every flag + outcome, and drift monitoring on the
  incoming slide distribution.

## Regulatory framing (important, not legal advice)

Framing PathQA as an internal **QA / workflow** tool that only surfaces cases for
human re-review — never rendering or altering a diagnosis — is a deliberately
conservative posture. Anything that outputs a diagnosis is a different, much
heavier regulatory category. Keep the "assist, not replace, human-in-the-loop"
boundary bright, and involve regulatory/clinical governance before any use on
real patient material. I'm not a lawyer or a regulatory advisor; validate the
intended-use claim with someone who is.

## Limitations

Binary tumor/normal patches only; no whole-slide aggregation yet; PatchCamelyon
is lymph-node metastasis and is not representative of your tissue types — the
public-data demo proves the *mechanism*, not clinical performance. Clinical
performance must be established on your own validated data.
```
