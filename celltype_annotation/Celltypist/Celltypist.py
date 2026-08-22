"""CellTypist reference-training cell type annotation benchmark."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import celltypist
import numpy as np
import pandas as pd
from scipy import sparse

from utils.dataset import normalize_log1p
from utils.pipeline import (
    SCORE_CLASSES_KEY,
    SCORE_KEY,
    create_argument_parser,
    run_annotation_pipeline,
    seed_python_numpy,
)


def _training_options(reference, seed):
    """Choose CellTypist's sparse-safe trainer for large references."""
    is_large = reference.n_obs > 100_000 and reference.n_vars > 10_000
    if not is_large:
        return {"random_state": seed}

    matrix_format = "sparse" if sparse.issparse(reference.X) else "dense"
    dense_gib = reference.n_obs * reference.n_vars * 8 / 1024**3
    print(
        "[Celltypist] large reference detected "
        f"({matrix_format}; dense equivalent ~{dense_gib:.1f} GiB); "
        "using sparse-safe mini-batch SGD"
    )
    return {
        "with_mean": False,
        "use_SGD": True,
        "mini_batch": True,
        "batch_number": 100,
        "batch_size": 1000,
        "epochs": 10,
        "n_jobs": 1,
        "random_state": seed,
    }


def train_celltypist(adata, seed, run_out_path, args):
    """Train CellTypist on log-normalized reference expression."""
    seed_python_numpy(seed)
    reference = normalize_log1p(adata, copy=False)
    model = celltypist.train(
        X=reference,
        labels=args.train_celltype,
        check_expression=True,
        **_training_options(reference, seed),
    )
    if args.save_models:
        model_dir = run_out_path / "_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.write(str(model_dir / "model.pkl"))
    return model


def _annotate_query(query, model, chunk_size=2048):
    """Run prediction in bounded-memory chunks."""
    if query.n_obs == 0:
        raise ValueError("Query dataset contains no cells")

    if query.n_obs > chunk_size:
        print(
            f"[Celltypist] predicting {query.n_obs} cells in chunks of "
            f"{chunk_size}"
        )

    predicted_parts = []
    probability_parts = []
    for start in range(0, query.n_obs, chunk_size):
        chunk = query[start : start + chunk_size].copy()
        if sparse.issparse(chunk.X):
            chunk.X = chunk.X.toarray()

        result = celltypist.annotate(filename=chunk, model=model)
        predicted_parts.append(result.predicted_labels.copy())
        probability_parts.append(result.probability_matrix.copy())

    return pd.concat(predicted_parts), pd.concat(probability_parts)


def predict_celltypist(adata, model, pred_key, seed, run_out_path, args):
    """Annotate query cells and retain CellTypist class probabilities."""
    query = normalize_log1p(adata, copy=False)
    predicted, probabilities = _annotate_query(query, model)

    predicted.index = predicted.index.astype(str)
    adata.obs[pred_key] = (
        predicted["predicted_labels"]
        .reindex(adata.obs_names.astype(str))
        .fillna("Unknown")
        .astype(str)
        .values
    )

    probabilities.index = probabilities.index.astype(str)
    probabilities = probabilities.reindex(adata.obs_names.astype(str))
    adata.obsm[SCORE_KEY] = probabilities.to_numpy(dtype=float)
    adata.uns[SCORE_CLASSES_KEY] = np.asarray(probabilities.columns, dtype=str)
    adata.uns["annotation_score_type"] = "probabilities"
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("Celltypist")
    arguments = parser.parse_args()
    run_annotation_pipeline(
        "Celltypist",
        "celltypist_prediction",
        train_celltypist,
        predict_celltypist,
        arguments,
    )
