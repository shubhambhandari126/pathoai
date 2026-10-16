"""QA session service — the stateful layer the app talks to.

Holds one review session in memory: the per-case reviews from the SecondReader,
the patch image for each case (so the UI can show it), and the pathologist's
decisions. Recomputes flags live when the QA threshold / workload changes, and
appends every decision to an audit log (governance requirement).

The model is loaded from a checkpoint if PATHQA_MODEL points at one; otherwise a
small demo model is fine-tuned on synthetic patches at first boot and cached, so
the app is runnable out of the box with nothing to set up.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from pathqa.data import (IMAGENET_MEAN, IMAGENET_STD, SyntheticPatches,
                         get_synthetic_loaders)
from pathqa.engine import fit_temperature, train_one_epoch
from pathqa.model import TemperatureScaler, build_model
from pathqa.qa import SecondReader
from pathqa.slide import SlideScorer, render_heatmap, review_slides
from pathqa.wsi import SlideTiler, SyntheticSlide

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def tensor_to_png(x: torch.Tensor) -> bytes:
    """De-normalize a CHW patch tensor back to a viewable PNG."""
    img = (x.detach().cpu() * _STD + _MEAN).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class QASession:
    def __init__(
        self,
        model_path: Optional[str] = None,
        audit_log: Optional[str] = None,
        tau: float = 0.9,
        budget: Optional[float] = None,
        n_cases: int = 200,
        path_error_rate: float = 0.08,
        demo_train_steps: int = 3,
        n_slides: int = 6,
        slide_grid: int = 14,
        slide_tau: float = 0.7,
    ):
        self.lock = threading.Lock()
        self.audit_log = audit_log or os.environ.get("PATHQA_AUDIT_LOG", "audit_log.jsonl")
        self.feedback_dir = os.environ.get("PATHQA_FEEDBACK_DIR", "data/feedback")
        # Images the bundled model already learned from: PathMNIST train (89,996)
        # + PatchCamelyon train (262,144). Pathologist submissions add on top.
        self.BASE_IMAGES = int(os.environ.get("PATHQA_BASE_IMAGES", "352140"))
        self.tau = tau
        self.budget = budget
        self.n_cases = n_cases
        self.path_error_rate = path_error_rate
        self.slide_tau = slide_tau
        self.device = torch.device("cpu")
        self.decisions: Dict[str, dict] = {}
        self.patches: Dict[str, torch.Tensor] = {}
        self.truth: Dict[str, int] = {}
        # slide state
        self.slide_scores: dict = {}
        self.slide_labels: Dict[str, int] = {}
        self.slide_truth: Dict[str, int] = {}
        self.slide_heatmaps: Dict[str, bytes] = {}
        self.slide_decisions: Dict[str, dict] = {}

        self.model, self.scaler = self._load_or_train(model_path, demo_train_steps)
        self.reader = SecondReader(self.model, self.scaler, self.device)
        self._build_signout()
        self.recompute()
        self._n_slides, self._slide_grid = n_slides, slide_grid
        if n_slides > 0:
            self._build_slides(n_slides, slide_grid)

    # ----- model -----
    def _load_or_train(self, model_path, steps):
        model = build_model("resnet18", pretrained=False)
        scaler = TemperatureScaler()
        # If no model given, use a bundled model.pt shipped next to the app.
        if not model_path:
            here = os.path.dirname(__file__)
            for cand in ("model.pt", os.path.join(here, "..", "model.pt")):
                if os.path.exists(cand):
                    model_path = cand; break
        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            scaler.set_temperature(float(ckpt.get("temperature", 1.0)))
            self.model_info = {"source": "trained", "metrics": ckpt.get("metrics", {}),
                               "file": Path(model_path).name}
            return model, scaler
        # Demo: quick fine-tune on synthetic patches so confidences are meaningful.
        train_dl, val_dl, _ = get_synthetic_loaders(n_train=256, n_eval=128, batch_size=32)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(max(1, steps)):
            train_one_epoch(model, train_dl, optim, self.device)
        scaler.set_temperature(fit_temperature(model, val_dl, self.device))
        self.model_info = {"source": "demo", "metrics": {}}
        return model, scaler

    # ----- data / sign-out queue -----
    def _build_signout(self):
        ds = SyntheticPatches(n=self.n_cases, seed=99)
        rng = torch.Generator().manual_seed(7)
        for i in range(len(ds)):
            x, true_label = ds[i]
            cid = f"case_{i:06d}"
            path_label = true_label
            if torch.rand(1, generator=rng).item() < self.path_error_rate:
                path_label = 1 - true_label  # simulated human miss
            self.patches[cid] = x
            self.truth[cid] = int(true_label)
            self.decisions.setdefault(cid, {"case_id": cid, "pathologist_label": int(path_label)})

    # ----- QA compute -----
    def recompute(self):
        with self.lock:
            ids = sorted(self.patches.keys())
            if not ids:  # no demo cohort (lean/production startup)
                from pathqa.qa import QAReport
                self.report = QAReport(reviews=[], tau=self.tau, n_cases=0,
                                       n_disagreements=0, n_flagged=0)
                self._by_id = {}
                return
            xb = torch.stack([self.patches[c] for c in ids])
            yb = torch.tensor([self.decisions[c]["pathologist_label"] for c in ids])

            class _One:
                def __iter__(self_):
                    yield xb, yb

            self.report = self.reader.review_loader(
                _One(), tau=self.tau, workload_budget=self.budget, case_ids=ids
            )
            self._by_id = {r.case_id: r for r in self.report.reviews}

    def set_threshold(self, tau: Optional[float] = None, budget: Optional[float] = None):
        if budget is not None:
            self.budget, self.tau = budget, 0.9
        elif tau is not None:
            self.tau, self.budget = tau, None
        self.recompute()

    # ----- views -----
    def summary(self) -> dict:
        r = self.report
        resolved = sum(1 for d in self.decisions.values() if d.get("action"))
        return {
            "n_cases": r.n_cases,
            "n_disagreements": r.n_disagreements,
            "n_flagged": r.n_flagged,
            "disagreement_rate": r.disagreement_rate,
            "flag_rate": r.flag_rate,
            "tau": r.tau,
            "mode": "budget" if self.budget is not None else "threshold",
            "budget": self.budget,
            "resolved": resolved,
        }

    def worklist(self) -> List[dict]:
        flagged = sorted(self.report.flagged, key=lambda r: r.model_confidence, reverse=True)
        out = []
        for i, r in enumerate(flagged, 1):
            d = self.decisions.get(r.case_id, {})
            out.append({
                "priority": i,
                "case_id": r.case_id,
                "pathologist_read": r.pathologist_read,
                "model_read": r.model_read,
                "model_confidence": round(r.model_confidence, 3),
                "model_prob_tumor": round(r.model_prob_tumor, 3),
                "reason": r.reason,
                "decision": d.get("action"),
                "note": d.get("note"),
            })
        return out

    def patch_png(self, case_id: str) -> Optional[bytes]:
        x = self.patches.get(case_id)
        return tensor_to_png(x) if x is not None else None

    def patch_cam_png(self, case_id: str) -> Optional[bytes]:
        """Grad-CAM overlay showing where the model looked on this patch."""
        x = self.patches.get(case_id)
        if x is None:
            return None
        from pathqa.gradcam import gradcam_png
        return gradcam_png(self.model, x, device=self.device)

    # ----- hot-swap the model from an uploaded checkpoint -----
    def load_model_from_upload(self, data_url: str) -> dict:
        import base64
        from io import BytesIO
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        ck = torch.load(BytesIO(raw), map_location="cpu")
        model = build_model(ck.get("backbone", "resnet18"), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        scaler = TemperatureScaler(); scaler.set_temperature(float(ck.get("temperature", 1.0)))
        with self.lock:
            self.model = model.to(self.device).eval()
            self.scaler = scaler
            self.reader = SecondReader(self.model, self.scaler, self.device)
            self.model_info = {"source": "uploaded", "metrics": ck.get("metrics", {})}
            self.recompute()
            if self._n_slides > 0:
                self.slide_scores, self.slide_labels, self.slide_truth = {}, {}, {}
                self.slide_heatmaps, self.slide_decisions = {}, {}
                self._build_slides(self._n_slides, self._slide_grid)
        return {"ok": True, "metrics": ck.get("metrics", {}), "labels": ck.get("labels", {}),
                "note": ck.get("note", "")}

    # ----- real image analysis (the trained CNN, not the browser heuristic) -----
    def analyze_image(self, data_url: str) -> dict:
        import base64
        from io import BytesIO
        from PIL import Image
        from pathqa.infer import analyze_image as _analyze
        b64 = data_url.split(",", 1)[-1]
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        res = _analyze(self.model, self.scaler, img, self.device)
        enc = lambda b: "data:image/png;base64," + base64.b64encode(b).decode()
        for src, dst in (("overlay_png", "overlay"), ("stain_overlay_png", "stain_overlay")):
            blob = res.pop(src, None)
            if blob is not None:
                res[dst] = enc(blob)
        for t in res.get("tiles", []):
            if "tile_png" in t:
                t["tile"] = enc(t.pop("tile_png"))
            if "cam_png" in t:
                t["cam"] = enc(t.pop("cam_png"))
        return res

    # ----- feedback: turn a reviewed image into training data -----
    def save_feedback(self, data_url: str, label: str, feedback_dir: Optional[str] = None,
                      auto_learn: bool = False, user: str = "anonymous") -> dict:
        import base64, time
        from io import BytesIO
        from PIL import Image
        feedback_dir = feedback_dir or self.feedback_dir
        if label not in ("normal", "tumor"):
            raise ValueError("label must be 'normal' or 'tumor'")
        d = Path(feedback_dir) / label
        d.mkdir(parents=True, exist_ok=True)
        img = Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[-1]))).convert("RGB")
        fn = d / f"{label}_{int(time.time()*1000)}.png"
        img.save(fn)
        # audit provenance: who submitted what, when
        try:
            with open(self.audit_log, "a") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "event": "feedback", "user": user,
                                    "label": label, "file": fn.name}) + "\n")
        except OSError:
            pass
        stats = self.feedback_stats(feedback_dir)
        out = {"stats": stats, "retrained": False}
        # Auto-learn: once there are a few of each label, fold every new
        # submission into the model and save it, so the next analysis uses it.
        if auto_learn and stats["normal"] >= 3 and stats["tumor"] >= 3:
            try:
                r = self.retrain_on_feedback(epochs=1)
                out["retrained"] = True
                out["val_auroc"] = r["val_auroc"]
                out["saved_to"] = r["saved_to"]
            except Exception as e:  # keep the submission even if training hiccups
                out["error"] = str(e)
        return out

    def retrain_on_feedback(self, epochs: int = 3) -> dict:
        """Continue training the CURRENT model on the labels submitted so far,
        save the updated model.pt, and hot-swap it in. This is the 'learn from
        pathologist submissions' loop, one click instead of a script."""
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets
        from pathqa.data import _eval_tfms, _train_tfms
        from pathqa.engine import evaluate as _eval

        root = Path(self.feedback_dir)
        counts = {c: len(list((root / c).glob("*.png"))) if (root / c).exists() else 0
                  for c in ("normal", "tumor")}
        if counts["normal"] < 3 or counts["tumor"] < 3:
            raise ValueError(f"Need at least 3 of each label first (have {counts}). Submit more cases.")

        full = datasets.ImageFolder(str(root))
        if full.class_to_idx != {"normal": 0, "tumor": 1}:
            raise ValueError(f"Unexpected label folders: {full.class_to_idx}")
        n = len(full); n_val = max(1, int(0.25 * n))
        g = torch.Generator().manual_seed(0)
        tr, va = random_split(full, [n - n_val, n_val], generator=g)
        tr.dataset = datasets.ImageFolder(str(root), transform=_train_tfms())
        va.dataset = datasets.ImageFolder(str(root), transform=_eval_tfms())
        train_dl = DataLoader(tr, batch_size=16, shuffle=True)
        val_dl = DataLoader(va, batch_size=16)

        with self.lock:
            optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            for _ in range(max(1, epochs)):
                train_one_epoch(self.model, train_dl, optim, self.device)
            res = _eval(self.model, val_dl, self.device)
            T = fit_temperature(self.model, val_dl, self.device)
            self.scaler.set_temperature(T)
            self.reader = SecondReader(self.model, self.scaler, self.device)
            # save to the model file in use (fall back to /tmp if read-only)
            save_to = os.environ.get("PATHQA_MODEL") or "model.pt"
            ckpt = {"state_dict": self.model.state_dict(), "backbone": "resnet18",
                    "temperature": T, "labels": {0: "normal", 1: "tumor"},
                    "metrics": {"val_auroc": res.auroc, "val_acc": res.accuracy},
                    "note": "Pathify model, continued training on submitted labels."}
            # Atomic write (write to .tmp then rename) so a crash mid-save can't
            # corrupt the live model.pt — important when it's on a GCS volume.
            def _atomic_save(path):
                tmp = path + ".tmp"
                torch.save(ckpt, tmp)
                os.replace(tmp, path)
            try:
                _atomic_save(save_to)
            except OSError:
                save_to = "/tmp/model.pt"; _atomic_save(save_to)
            self.model_info = {"source": "retrained",
                               "metrics": {"test_auroc": res.auroc, "val_acc": res.accuracy}}
            self.recompute()
            if self._n_slides > 0:
                self.slide_scores, self.slide_labels, self.slide_truth = {}, {}, {}
                self.slide_heatmaps, self.slide_decisions = {}, {}
                self._build_slides(self._n_slides, self._slide_grid)
        return {"ok": True, "n_train": n - n_val, "n_val": n_val,
                "val_auroc": round(res.auroc, 3), "saved_to": save_to, "counts": counts}

    def feedback_stats(self, feedback_dir: Optional[str] = None) -> dict:
        base = Path(feedback_dir or self.feedback_dir)
        counts = {}
        for lab in ("normal", "tumor"):
            p = base / lab
            counts[lab] = len(list(p.glob("*.png"))) if p.exists() else 0
        counts["total"] = counts["normal"] + counts["tumor"]
        counts["ready_to_train"] = counts["normal"] >= 10 and counts["tumor"] >= 10
        return counts

    # ----- slides -----
    def _build_slides(self, n_slides: int, grid: int):
        import random as _r
        rng = _r.Random(11)
        scorer = SlideScorer(self.model, self.scaler, self.device, topk_frac=0.02, min_topk=4)
        for i in range(n_slides):
            has_tumor = (i % 2 == 0)
            slide = SyntheticSlide(grid=grid, tumor=has_tumor, seed=100 + i)
            sid = f"slide_{i:03d}"
            tiler = SlideTiler(slide.image(), tile_size=slide.tile, model_size=96)
            sc = scorer.score_slide(sid, tiler)
            lb = int(has_tumor)
            if rng.random() < 0.25:
                lb = 1 - lb  # simulated sign-out error
            self.slide_scores[sid] = sc
            self.slide_truth[sid] = int(has_tumor)
            self.slide_labels[sid] = lb
            self.slide_heatmaps[sid] = render_heatmap(sc, base_image=slide.image())
            self.slide_decisions.setdefault(sid, {"slide_id": sid, "pathologist_label": lb})

    def slide_summary(self) -> dict:
        reviews = review_slides(self.slide_scores, self.slide_labels, tau=self.slide_tau)
        flagged = sum(1 for r in reviews if r.flagged)
        resolved = sum(1 for d in self.slide_decisions.values() if d.get("action"))
        return {"n_slides": len(reviews), "n_flagged": flagged,
                "resolved": resolved, "slide_tau": self.slide_tau}

    def slide_worklist(self) -> List[dict]:
        reviews = review_slides(self.slide_scores, self.slide_labels, tau=self.slide_tau)
        flagged = sorted([r for r in reviews if r.flagged],
                         key=lambda r: r.confidence, reverse=True)
        out = []
        for i, r in enumerate(flagged, 1):
            sc = self.slide_scores[r.slide_id]
            d = self.slide_decisions.get(r.slide_id, {})
            out.append({
                "priority": i, "slide_id": r.slide_id,
                "pathologist_read": r.pathologist_read, "model_read": r.model_read,
                "slide_score": round(r.slide_score, 3), "confidence": round(r.confidence, 3),
                "n_tumor_tiles": r.n_tumor_tiles, "n_tissue_tiles": sc.n_tissue_tiles,
                "reason": r.reason, "decision": d.get("action"),
            })
        return out

    def slide_heatmap_png(self, slide_id: str) -> Optional[bytes]:
        return self.slide_heatmaps.get(slide_id)

    def set_slide_tau(self, tau: float):
        self.slide_tau = float(tau)

    def record_slide_decision(self, slide_id: str, action: str, note: str = "", user: str = "anonymous") -> dict:
        if slide_id not in self.slide_decisions:
            raise KeyError(slide_id)
        if action not in ("confirm_signout", "amend"):
            raise ValueError("action must be 'confirm_signout' or 'amend'")
        sc = self.slide_scores.get(slide_id)
        with self.lock:
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": "slide", "user": user,
                "slide_id": slide_id, "action": action, "note": note,
                "pathologist_label": self.slide_decisions[slide_id]["pathologist_label"],
                "model_read": sc.model_read if sc else None,
                "slide_score": round(sc.slide_score, 3) if sc else None,
            }
            self.slide_decisions[slide_id].update({"action": action, "note": note})
            with open(self.audit_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

    def record_decision(self, case_id: str, action: str, note: str = "", user: str = "anonymous") -> dict:
        if case_id not in self.decisions:
            raise KeyError(case_id)
        if action not in ("confirm_signout", "amend"):
            raise ValueError("action must be 'confirm_signout' or 'amend'")
        with self.lock:
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": user,
                "case_id": case_id,
                "action": action,           # confirm_signout = pathologist kept their read
                                            # amend = re-review changed the read (model caught it)
                "note": note,
                "pathologist_label": self.decisions[case_id]["pathologist_label"],
                "model_read": self._by_id[case_id].model_read if case_id in self._by_id else None,
                "model_confidence": (round(self._by_id[case_id].model_confidence, 3)
                                     if case_id in self._by_id else None),
            }
            self.decisions[case_id].update({"action": action, "note": note})
            with open(self.audit_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry
