"""Second-reader QA engine — the safety net.

The design principle: this tool NEVER makes a diagnosis and NEVER auto-changes a
pathologist's call. It runs silently behind the sign-out, and its only action is
to raise a hand: "case N — worth a second look, here's why." A human decides.

Workflow per case (a patch, or an aggregated slide):
    pathologist_label  : the human's read (0 normal / 1 tumor)
    model_prob         : calibrated P(tumor) from the classifier
    model_pred         : argmax of the model
    agree              : model_pred == pathologist_label
    model_confidence   : max(model_prob, 1 - model_prob)  in [0.5, 1.0]

A case is FLAGGED for re-check when the model DISAGREES with the pathologist and
does so CONFIDENTLY (confidence >= tau). Two ways to set the bar:

  * fixed threshold `tau`  — flag every confident disagreement, or
  * `workload_budget`      — flag only the top X% most-confident disagreements,
                             so the QA queue fits the time the lab actually has.

The output is a prioritized worklist, sorted by model confidence, plus summary
stats (flag rate, disagreement rate). That is the entire product surface.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class CaseReview:
    case_id: str
    pathologist_label: int          # 0 normal, 1 tumor
    model_pred: int                 # 0 normal, 1 tumor
    model_prob_tumor: float         # calibrated P(tumor)
    model_confidence: float         # in [0.5, 1.0]
    agree: bool
    flagged: bool
    reason: str

    @property
    def pathologist_read(self) -> str:
        return "tumor" if self.pathologist_label == 1 else "normal"

    @property
    def model_read(self) -> str:
        return "tumor" if self.model_pred == 1 else "normal"


@dataclass
class QAReport:
    reviews: List[CaseReview]
    tau: float
    n_cases: int
    n_disagreements: int
    n_flagged: int

    @property
    def disagreement_rate(self) -> float:
        return self.n_disagreements / max(self.n_cases, 1)

    @property
    def flag_rate(self) -> float:
        return self.n_flagged / max(self.n_cases, 1)

    @property
    def flagged(self) -> List[CaseReview]:
        return [r for r in self.reviews if r.flagged]

    def summary(self) -> str:
        return (
            f"PathQA second-reader report\n"
            f"  cases reviewed        : {self.n_cases}\n"
            f"  model/pathologist disagreements: {self.n_disagreements} "
            f"({self.disagreement_rate:.1%})\n"
            f"  flagged for re-check  : {self.n_flagged} ({self.flag_rate:.1%}) "
            f"at confidence >= {self.tau:.2f}\n"
            f"  -> {self.n_cases - self.n_flagged} cases pass QA untouched; "
            f"pathologist reviews only the {self.n_flagged} flagged."
        )

    def to_json(self, path: str) -> None:
        payload = {
            "tau": self.tau,
            "n_cases": self.n_cases,
            "n_disagreements": self.n_disagreements,
            "n_flagged": self.n_flagged,
            "disagreement_rate": self.disagreement_rate,
            "flag_rate": self.flag_rate,
            "reviews": [asdict(r) for r in self.reviews],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def worklist_to_csv(self, path: str) -> None:
        """Write ONLY the flagged cases, highest confidence first — the queue a
        pathologist actually opens."""
        rows = sorted(self.flagged, key=lambda r: r.model_confidence, reverse=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["priority", "case_id", "pathologist_read", "model_read",
                        "model_confidence", "reason"])
            for i, r in enumerate(rows, 1):
                w.writerow([i, r.case_id, r.pathologist_read, r.model_read,
                            f"{r.model_confidence:.3f}", r.reason])


class SecondReader:
    """Runs a trained, calibrated classifier as a QA second reader."""

    def __init__(self, model, scaler=None, device: Optional[torch.device] = None):
        self.model = model
        self.scaler = scaler
        self.device = device or torch.device("cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x.to(self.device))
        if self.scaler is not None:
            logits = self.scaler(logits)
        return F.softmax(logits, dim=1)[:, 1].cpu()  # P(tumor)

    @torch.no_grad()
    def review_loader(
        self,
        loader: DataLoader,
        tau: float = 0.9,
        workload_budget: Optional[float] = None,
        case_ids: Optional[Sequence[str]] = None,
        max_batches: Optional[int] = None,
    ) -> QAReport:
        """Review a labeled set. The loader must yield (image, pathologist_label).

        Args:
            tau: confidence threshold for flagging a disagreement (ignored if
                workload_budget is set).
            workload_budget: if set (e.g. 0.05), flag only the top 5% of
                disagreements by confidence, and back out the effective tau.
        """
        probs_all, labels_all = [], []
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            probs_all.append(self._probs(x))
            labels_all.append(y)
        probs = torch.cat(probs_all)
        labels = torch.cat(labels_all).long()
        ids = list(case_ids) if case_ids is not None else [f"case_{i:06d}" for i in range(len(labels))]
        return self._build_report(ids, probs, labels, tau, workload_budget)

    @staticmethod
    def _build_report(ids, probs, labels, tau, workload_budget) -> QAReport:
        preds = (probs >= 0.5).long()
        confidence = torch.maximum(probs, 1 - probs)
        disagree = preds != labels

        # If a workload budget is given, set tau to flag only the top-X% most
        # confident disagreements.
        if workload_budget is not None:
            dis_conf = confidence[disagree]
            if len(dis_conf) == 0:
                tau = 1.01  # nothing to flag
            else:
                k = max(1, int(round(workload_budget * len(labels))))
                k = min(k, len(dis_conf))
                tau = float(torch.sort(dis_conf, descending=True).values[k - 1])

        reviews: List[CaseReview] = []
        for cid, p, lb, pr, cf, dg in zip(
            ids, probs.tolist(), labels.tolist(), preds.tolist(),
            confidence.tolist(), disagree.tolist()
        ):
            flagged = bool(dg and cf >= tau)
            if flagged:
                reason = (
                    f"Model reads {'tumor' if pr == 1 else 'normal'} "
                    f"(confidence {cf:.2f}) but sign-out is "
                    f"{'tumor' if lb == 1 else 'normal'} — recommend re-review."
                )
            elif dg:
                reason = "Model disagrees but below confidence threshold — not flagged."
            else:
                reason = "Model agrees with pathologist."
            reviews.append(CaseReview(
                case_id=cid, pathologist_label=int(lb), model_pred=int(pr),
                model_prob_tumor=float(p), model_confidence=float(cf),
                agree=not bool(dg), flagged=flagged, reason=reason,
            ))

        return QAReport(
            reviews=reviews,
            tau=float(tau),
            n_cases=len(reviews),
            n_disagreements=int(disagree.sum()),
            n_flagged=sum(r.flagged for r in reviews),
        )
