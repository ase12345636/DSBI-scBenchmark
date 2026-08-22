"""Aggregate and plot results from the unified annotation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUALITY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "auc_ovr_macro",
]
RESOURCE_METRICS = [
    "runtime_seconds",
    "peak_ram_mb",
    "peak_vram_mb",
]


def collect_overall_metrics(results_dir):
    """Read every method-level metrics_summary.csv below results_dir."""
    frames = []
    for path in sorted(Path(results_dir).glob("*/metrics_summary.csv")):
        frame = pd.read_csv(path)
        if "tool" not in frame:
            frame["tool"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_per_class_metrics(results_dir):
    """Read all per-run per_class_metrics.csv files."""
    rows = []
    for method_dir in sorted(path for path in Path(results_dir).iterdir() if path.is_dir()):
        for path in sorted(method_dir.glob("*/per_class_metrics.csv")):
            frame = pd.read_csv(path)
            frame.insert(0, "run_id", path.parent.name)
            frame.insert(0, "tool", method_dir.name)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_metrics(frame, group_columns):
    """Return mean/std columns for every numeric benchmark metric."""
    if frame.empty:
        return pd.DataFrame()
    numeric_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in group_columns and column not in {"run_id", "seed"}
    ]
    summary = frame.groupby(group_columns, dropna=False)[numeric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def plot_metric_group(summary, metrics, output_path, title):
    """Draw grouped method bars with standard-deviation error bars."""
    available = [metric for metric in metrics if f"{metric}_mean" in summary]
    if summary.empty or not available:
        return

    methods = summary["tool"].astype(str).tolist()
    positions = np.arange(len(methods), dtype=float)
    width = 0.8 / len(available)
    fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.4), 6))

    for index, metric in enumerate(available):
        means = pd.to_numeric(summary[f"{metric}_mean"], errors="coerce")
        std_column = f"{metric}_std"
        errors = (
            pd.to_numeric(summary[std_column], errors="coerce").fillna(0)
            if std_column in summary else None
        )
        offset = (index - (len(available) - 1) / 2) * width
        ax.bar(
            positions + offset,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=metric,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Score" if "quality" in title.lower() else "Value")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_results(results_dir, output_dir=None):
    """Create cross-method CSV summaries and comparison figures."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = collect_overall_metrics(results_dir)
    if overall.empty:
        raise FileNotFoundError(
            f"No METHOD/metrics_summary.csv files found below {results_dir}"
        )
    overall.to_csv(output_dir / "all_runs_metrics.csv", index=False)
    overall_summary = summarize_metrics(overall, ["tool"])
    overall_summary.to_csv(output_dir / "method_metrics_summary.csv", index=False)

    per_class = collect_per_class_metrics(results_dir)
    if not per_class.empty:
        per_class.to_csv(output_dir / "all_runs_per_class_metrics.csv", index=False)
        summarize_metrics(per_class, ["tool", "celltype"]).to_csv(
            output_dir / "method_per_class_summary.csv", index=False
        )

    plot_metric_group(
        overall_summary,
        QUALITY_METRICS,
        output_dir / "annotation_quality_comparison.png",
        "Cell type annotation quality",
    )
    plot_metric_group(
        overall_summary,
        RESOURCE_METRICS,
        output_dir / "resource_comparison.png",
        "Runtime and resource comparison",
    )
    print(f"Analysis saved to {output_dir}")
    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize annotation benchmark results")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    arguments = parser.parse_args()
    analyze_results(arguments.results_dir, arguments.output_dir)
