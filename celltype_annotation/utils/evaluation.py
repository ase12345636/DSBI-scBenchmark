"""Common classification metrics for cell type annotation benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _clean_labels(values, missing_value="Unknown"):
    labels = pd.Series(values, dtype="string").str.strip()
    return labels.fillna(missing_value).replace("", missing_value).astype(str)


def _extract_scores(adata, pred_key, score_key, score_classes_key):
    """Return score matrix, class names and an honest score-source label."""
    if score_key in adata.obsm and score_classes_key in adata.uns:
        scores = np.asarray(adata.obsm[score_key], dtype=float)
        classes = np.asarray(adata.uns[score_classes_key]).astype(str)
        if scores.ndim != 2 or scores.shape != (adata.n_obs, len(classes)):
            raise ValueError(
                f"{score_key} must have shape (n_cells, n_score_classes)"
            )
        source = str(adata.uns.get("annotation_score_type", "continuous_scores"))
        return scores, classes, source

    pred = _clean_labels(adata.obs[pred_key])
    classes = np.asarray(sorted(pred.unique()), dtype=str)
    class_to_index = {label: index for index, label in enumerate(classes)}
    scores = np.zeros((adata.n_obs, len(classes)), dtype=float)
    for row, label in enumerate(pred):
        scores[row, class_to_index[label]] = 1.0
    return scores, classes, "hard_predictions"


def _auc_metrics(y_true, scores, score_classes):
    """Calculate one-vs-rest AUCs without requiring identical class sets."""
    class_to_index = {label: index for index, label in enumerate(score_classes)}
    per_class_auc = {}
    supports = {}

    for label in sorted(set(y_true)):
        binary_true = (y_true == label).astype(int)
        supports[label] = int(binary_true.sum())
        if label not in class_to_index or binary_true.min() == binary_true.max():
            per_class_auc[label] = np.nan
            continue
        try:
            per_class_auc[label] = float(
                roc_auc_score(binary_true, scores[:, class_to_index[label]])
            )
        except ValueError:
            per_class_auc[label] = np.nan

    valid = {key: value for key, value in per_class_auc.items() if np.isfinite(value)}
    if not valid:
        return np.nan, np.nan, per_class_auc

    auc_macro = float(np.mean(list(valid.values())))
    total_support = sum(supports[label] for label in valid)
    auc_weighted = float(
        sum(valid[label] * supports[label] for label in valid) / total_support
    )
    return auc_macro, auc_weighted, per_class_auc


def evaluate_predictions(
    adata,
    pred_key,
    celltype_key,
    score_key="annotation_scores",
    score_classes_key="annotation_score_classes",
):
    """Evaluate predictions and return overall and per-class metric tables."""
    if celltype_key not in adata.obs:
        raise KeyError(f"Ground-truth column '{celltype_key}' not found in test data")
    if pred_key not in adata.obs:
        raise KeyError(f"Prediction column '{pred_key}' not found")

    raw_true = pd.Series(adata.obs[celltype_key], index=adata.obs_names, dtype="string")
    valid = raw_true.notna() & raw_true.str.strip().ne("")
    if not valid.any():
        raise ValueError("The test data contains no non-missing ground-truth labels")

    y_true = _clean_labels(raw_true[valid]).to_numpy()
    y_pred = _clean_labels(adata.obs.loc[valid, pred_key]).to_numpy()
    scores, score_classes, auc_source = _extract_scores(
        adata, pred_key, score_key, score_classes_key
    )
    scores = scores[np.asarray(valid)]

    labels = np.asarray(sorted(set(y_true) | set(y_pred)), dtype=str)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )
    auc_macro, auc_weighted, per_class_auc = _auc_metrics(
        y_true, scores, score_classes
    )

    overall = {
        "n_test_cells": int(len(y_true)),
        "n_true_classes": int(len(set(y_true))),
        "n_predicted_classes": int(len(set(y_pred))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "precision_micro": float(precision_micro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_micro": float(recall_micro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "auc_ovr_macro": auc_macro,
        "auc_ovr_weighted": auc_weighted,
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "unknown_prediction_rate": float(np.mean(y_pred == "Unknown")),
        "auc_score_source": auc_source,
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_rows = []
    for index, label in enumerate(labels):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        per_class_rows.append({
            "celltype": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "specificity": float(specificity),
            "f1": float(f1[index]),
            "auc_ovr": per_class_auc.get(label, np.nan),
            "support": int(support[index]),
        })

    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(cm, index=labels, columns=labels)
    return overall, per_class, confusion
