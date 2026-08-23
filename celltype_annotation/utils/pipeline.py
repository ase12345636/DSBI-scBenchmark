"""Unified train/predict/evaluate pipeline for annotation benchmarks.

Each tool implements only model training and prediction. Loading, repeated
runs, metrics, UMAPs, result files and resource tracking are shared here, in
the same style as ``batch_effect/utils/pipeline.py``.
"""

from __future__ import annotations

import argparse
import gc
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csc_matrix, issparse

from utils.dataset import load_dataset
from utils.evaluation import evaluate_predictions
from utils.function import MemoryTracker, compute_and_save_umap, save_confusion_heatmap

SCORE_KEY = "annotation_scores"
SCORE_CLASSES_KEY = "annotation_score_classes"


# ============================================================
# Argument Parsing
# ============================================================

def optional_key(value):
    """Convert CLI None/null/empty values into Python None."""
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in {"", "none", "null"} else value


def create_argument_parser(tool_name):
    """Create the standard CLI shared by every annotation method."""
    parser = argparse.ArgumentParser(description=f"Run {tool_name} cell type annotation")
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--train_celltype", type=optional_key, default="celltype")
    parser.add_argument("--test_celltype", type=optional_key, default="celltype")
    parser.add_argument("--train_batch", type=optional_key, default="batch")
    parser.add_argument("--test_batch", type=optional_key, default="batch")
    parser.add_argument(
        "--run_times", "--n_runs", dest="run_times", type=int, default=1
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help="Use base_seed + run_id - 1; default generates fresh seeds",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Keep per-run trained model files",
    )
    return parser


def seed_python_numpy(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate_seed(run_id, base_seed=None):
    if base_seed is not None:
        return int(base_seed + run_id - 1)
    return random.SystemRandom().randrange(1, 2**31 - 1)


# ============================================================
# Python ↔ R Exchange
# ============================================================

def _rscript_executable():
    """Prefer Rscript from the active Python/Conda environment."""
    env_rscript = Path(sys.executable).with_name("Rscript")
    if env_rscript.is_file():
        return str(env_rscript)
    discovered = shutil.which("Rscript")
    if discovered is None:
        raise FileNotFoundError("Rscript was not found in the active environment or PATH")
    return discovered


def export_adata_for_r(adata, output_dir, layer=None):
    """Export genes × cells Matrix Market data and matching metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if not issparse(matrix):
        matrix = csc_matrix(matrix)
    mmwrite(output_dir / "matrix.mtx", matrix.T)
    pd.DataFrame({"barcode": adata.obs_names.astype(str)}).to_csv(
        output_dir / "barcodes.csv", index=False
    )
    pd.DataFrame({"feature": adata.var_names.astype(str)}).to_csv(
        output_dir / "features.csv", index=False
    )
    adata.obs.to_csv(output_dir / "metadata.csv")


def train_r_reference(
    adata, seed, run_out_path, args, r_script, extra_args=None, layer=None
):
    """Train and serialize an R reference model for one run."""
    model_dir = Path(run_out_path) / "_model"
    exchange_dir = model_dir / "train_exchange"
    export_adata_for_r(adata, exchange_dir, layer=layer)
    command = [
        _rscript_executable(), str(r_script),
        "--task", "train_reference",
        "--input_dir", str(exchange_dir),
        "--save_path", str(model_dir),
        "--label_col", args.train_celltype,
        "--seed", str(seed),
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)
    shutil.rmtree(exchange_dir)
    return model_dir / "reference.rds"


def predict_r_reference(
    adata,
    model_path,
    pred_key,
    seed,
    run_out_path,
    args,
    r_script,
    score_type="continuous_scores",
    extra_args=None,
    layer=None,
):
    """Run an R predictor and import labels plus optional class scores."""
    predict_dir = Path(run_out_path) / "_r_predict"
    exchange_dir = predict_dir / "test_exchange"
    export_adata_for_r(adata, exchange_dir, layer=layer)
    command = [
        _rscript_executable(), str(r_script),
        "--task", "predict",
        "--input_dir", str(exchange_dir),
        "--reference_rds", str(model_path),
        "--save_path", str(predict_dir),
        "--label_col", args.train_celltype,
        "--seed", str(seed),
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)

    pred_frame = pd.read_csv(predict_dir / "predictions.csv", dtype={"barcode": str})
    if not {"barcode", "prediction"}.issubset(pred_frame.columns):
        raise ValueError("R predictions.csv must contain barcode and prediction columns")
    pred = pred_frame.set_index("barcode")["prediction"]
    adata.obs[pred_key] = pred.reindex(adata.obs_names.astype(str)).fillna("Unknown").values

    score_path = predict_dir / "scores.csv"
    if score_path.exists():
        score_frame = pd.read_csv(score_path, dtype={"barcode": str})
        if "barcode" not in score_frame:
            raise ValueError("R scores.csv must contain a barcode column")
        score_frame = score_frame.set_index("barcode").reindex(adata.obs_names.astype(str))
        adata.obsm[SCORE_KEY] = score_frame.to_numpy(dtype=float)
        adata.uns[SCORE_CLASSES_KEY] = np.asarray(score_frame.columns, dtype=str)
        adata.uns["annotation_score_type"] = score_type

    shutil.rmtree(predict_dir)
    return adata


# ============================================================
# Result Writers
# ============================================================

def _save_predictions(adata, pred_key, test_batch, test_celltype, output_dir):
    frame = pd.DataFrame(index=adata.obs_names)
    frame.index.name = "barcode"
    frame["prediction"] = adata.obs[pred_key].astype(str)
    if test_celltype is not None:
        frame["celltype"] = adata.obs[test_celltype].astype(str)
    if test_batch is not None:
        frame["batch"] = adata.obs[test_batch].astype(str)
    if SCORE_KEY in adata.obsm:
        scores = np.asarray(adata.obsm[SCORE_KEY], dtype=float)
        if scores.shape[1] > 0:
            frame["prediction_score"] = np.nanmax(scores, axis=1)
    frame.to_csv(Path(output_dir) / "predictions.csv")


def _write_run_results(
    adata, metrics, per_class, confusion, tool_name, pred_key, args, seed,
    run_out_path,
):
    run_out_path = Path(run_out_path)
    pd.DataFrame([metrics]).to_csv(
        run_out_path / f"{tool_name}_metrics.csv", index=False
    )
    _save_predictions(
        adata, pred_key, args.test_batch, args.test_celltype, run_out_path
    )

    if per_class is not None and confusion is not None:
        per_class.to_csv(run_out_path / "per_class_metrics.csv", index=False)
        confusion.to_csv(run_out_path / "confusion_matrix.csv")
        save_confusion_heatmap(
            confusion,
            run_out_path / f"{tool_name}_confusion_matrix.png",
            f"{tool_name} prediction",
        )

    compute_and_save_umap(
        adata,
        pred_key,
        args.test_batch,
        args.test_celltype,
        seed,
        run_out_path,
        tool_name,
    )


# ============================================================
# Main Pipeline
# ============================================================

def _validate_key(adata, key, dataset_name, role):
    if key is not None and key not in adata.obs:
        raise KeyError(f"{dataset_name} {role} column '{key}' not found")


def run_annotation_pipeline(tool_name, pred_key, train_fn, predict_fn, args):
    """Run repeated reference training, query prediction and evaluation."""
    if args.run_times < 1:
        raise ValueError("--n_runs/--run_times must be at least 1")

    for name in ["train_celltype", "test_celltype", "train_batch", "test_batch"]:
        setattr(args, name, optional_key(getattr(args, name, None)))

    args.save_path.mkdir(parents=True, exist_ok=True)
    train_raw = load_dataset(args.train_path)
    test_raw = load_dataset(args.test_path)

    _validate_key(train_raw, args.train_celltype, "Training", "cell-type")
    _validate_key(test_raw, args.test_celltype, "Test", "cell-type")
    _validate_key(train_raw, args.train_batch, "Training", "batch")
    _validate_key(test_raw, args.test_batch, "Test", "batch")

    if args.train_celltype is None:
        raise ValueError(
            "--train_celltype=None means no training labels, but all methods in "
            "this supervised benchmark require labelled reference cells"
        )

    valid_train = train_raw.obs[args.train_celltype].notna()
    valid_train &= train_raw.obs[args.train_celltype].astype(str).str.strip().ne("")
    if not valid_train.all():
        print(f"[train] dropping {int((~valid_train).sum())} cells with missing labels")
        train_raw = train_raw[valid_train].copy()
    if train_raw.obs[args.train_celltype].nunique() < 2:
        raise ValueError("Training data must contain at least two cell types")

    print(
        f"Train: {train_raw.n_obs} cells × {train_raw.n_vars} genes\n"
        f"Test:  {test_raw.n_obs} cells × {test_raw.n_vars} genes\n"
        f"Train celltype: {args.train_celltype} | Test celltype: {args.test_celltype}\n"
        f"Train batch: {args.train_batch} | Test batch: {args.test_batch}"
    )
    if args.test_celltype is None:
        print("[evaluation] test_celltype=None: classification metrics are skipped")

    summary_rows = []
    tool_root = args.save_path / tool_name
    for run_id in range(1, args.run_times + 1):
        seed = generate_seed(run_id, args.base_seed)
        seed_python_numpy(seed)
        run_out_path = tool_root / str(run_id)
        run_out_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"{tool_name} run {run_id}/{args.run_times} | seed={seed}")
        print(f"{'=' * 60}\n")

        tracker = MemoryTracker()
        tracker.start()
        try:
            train_start = time.time()
            model_handle = train_fn(train_raw.copy(), seed, run_out_path, args)
            train_runtime = time.time() - train_start

            predict_start = time.time()
            annotated = predict_fn(
                test_raw.copy(), model_handle, pred_key, seed, run_out_path, args
            )
            predict_runtime = time.time() - predict_start
        except Exception:
            tracker.stop()
            raise
        memory = tracker.stop()

        if list(annotated.obs_names) != list(test_raw.obs_names):
            if set(annotated.obs_names) != set(test_raw.obs_names):
                raise ValueError("Prediction output cells do not match the test dataset")
            annotated = annotated[test_raw.obs_names].copy()
        if pred_key not in annotated.obs:
            raise KeyError(f"Method did not create prediction column '{pred_key}'")
        annotated.obs[pred_key] = (
            annotated.obs[pred_key].astype("string").fillna("Unknown").astype(str)
        )

        if args.test_celltype is not None:
            metrics, per_class, confusion = evaluate_predictions(
                annotated, pred_key, args.test_celltype
            )
            evaluation_status = "computed"
        else:
            metrics, per_class, confusion = {}, None, None
            evaluation_status = "skipped_no_test_celltype"

        metrics.update({
            "tool": tool_name,
            "run_id": run_id,
            "seed": seed,
            "evaluation_status": evaluation_status,
            "train_runtime_seconds": train_runtime,
            "predict_runtime_seconds": predict_runtime,
            "runtime_seconds": train_runtime + predict_runtime,
            "peak_ram_mb": memory["peak_ram_mb"],
            "peak_vram_mb": memory["peak_vram_mb"],
        })
        _write_run_results(
            annotated, metrics, per_class, confusion, tool_name, pred_key,
            args, seed, run_out_path,
        )
        summary_rows.append(metrics)

        if not args.save_models:
            model_dir = run_out_path / "_model"
            if model_dir.exists():
                shutil.rmtree(model_dir)
        del model_handle, annotated
        gc.collect()

        if evaluation_status == "computed":
            print(
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"F1(macro)={metrics['f1_macro']:.4f}, "
                f"AUC(macro)={metrics['auc_ovr_macro']:.4f}"
            )
        print(f"Results saved to {run_out_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tool_root / "metrics_summary.csv", index=False)
    numeric = summary.select_dtypes(include=[np.number]).columns
    aggregate = summary[numeric].agg(["mean", "std"]).T
    aggregate.index.name = "metric"
    aggregate.to_csv(tool_root / "metrics_aggregate.csv")
    print(f"\nAll {tool_name} runs completed. Results saved to {tool_root}")
