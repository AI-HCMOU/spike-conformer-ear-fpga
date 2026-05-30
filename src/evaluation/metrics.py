"""
Evaluation metrics: Rank-1, Rank-5, F1, Precision, Recall, MCC, AUC.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)
from typing import Dict, List


class MetricTracker:
    """Accumulates predictions and computes all paper metrics."""

    def __init__(self, metric_names: List[str]):
        self.metric_names = [m.lower() for m in metric_names]
        self.reset()

    def reset(self):
        self.all_preds: list = []
        self.all_targets: list = []
        self.all_probs: list = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Accumulate batch predictions."""
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute all requested metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)
        results = {}

        for name in self.metric_names:
            match name:
                case "rank1" | "accuracy":
                    results["rank1"] = accuracy_score(targets, preds)
                case "rank5":
                    results["rank5"] = self._rank_k(probs, targets, k=5)
                case "precision":
                    results["precision"] = precision_score(
                        targets, preds, average="macro", zero_division=0
                    )
                case "recall":
                    results["recall"] = recall_score(
                        targets, preds, average="macro", zero_division=0
                    )
                case "f1":
                    results["f1"] = f1_score(
                        targets, preds, average="macro", zero_division=0
                    )
                case "mcc":
                    results["mcc"] = matthews_corrcoef(targets, preds)
                case "auc":
                    try:
                        results["auc"] = roc_auc_score(
                            targets, probs, multi_class="ovr", average="macro"
                        )
                    except ValueError:
                        results["auc"] = 0.0

        return results

    @staticmethod
    def _rank_k(probs: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
        """Rank-k accuracy: correct class in top-k predictions."""
        top_k = np.argsort(probs, axis=1)[:, -k:]
        correct = sum(t in top_k[i] for i, t in enumerate(targets))
        return correct / len(targets)
