# -*- coding: utf-8 -*-
"""
Utility functions for evaluating classification models.

This module provides a helper to evaluate a PyTorch classification model on a
validation/test DataLoader, compute common metrics, and optionally save the
results to an Excel file.

Example
-------
from evaluation import evaluate_classification_model

metrics = evaluate_classification_model(
    model,
    dataloader_val,
    device="cuda",
    save_dir="results",
    save_excel=True,
)
print(metrics["weighted_accuracy"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

__all__ = ["evaluate_classification_model"]


def evaluate_classification_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    save_dir: Optional[Union[str, Path]] = None,
    save_excel: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a classification model on a given DataLoader.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model to evaluate.
    dataloader : DataLoader
        DataLoader providing (inputs, labels) batches.
    device : str or torch.device, optional
        Device on which to run the evaluation (e.g., "cuda", "cpu").
        Default is "cpu".
    save_dir : str or pathlib.Path, optional
        Directory where evaluation results will be saved if `save_excel` is True.
        If None and `save_excel` is True, results will not be saved.
    save_excel : bool, optional
        If True, save evaluation results to an Excel file in `save_dir`.
        Default is False.

    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluation metrics:
        - "accuracy": float
        - "weighted_accuracy": float
        - "classification_report": dict (from sklearn)
        - "confusion_matrix": np.ndarray
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Apply log_softmax + argmax (equivalent to argmax on logits)
            y_pred_softmax = torch.log_softmax(outputs, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(y_pred_tags.cpu().tolist())

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Frequency-weighted accuracy (preserving your original logic, cleaned up)
    class_freq = cm.sum(axis=1).astype(float)
    total = cm.sum()
    class_weights = class_freq / total  # per-class frequency
    # class_weights.sum() == 1, so denominator would be 1
    weighted_acc = float((cm.diagonal() * class_weights).sum())

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "weighted_accuracy": weighted_acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # Optional Excel export
    if save_excel and save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        excel_path = save_dir / "evaluation_results.xlsx"

        # Summary sheet
        summary_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Weighted_Accuracy"],
                "Value": [acc, weighted_acc],
            }
        )

        # Classification report sheet
        report_df = pd.DataFrame(report).transpose()

        # Confusion matrix sheet
        cm_df = pd.DataFrame(cm)

        with pd.ExcelWriter(excel_path) as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            report_df.to_excel(writer, sheet_name="classification_report")
            cm_df.to_excel(writer, sheet_name="confusion_matrix", index=False)

    return metrics
