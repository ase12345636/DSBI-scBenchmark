"""CellTypist reference-training cell type annotation benchmark."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import celltypist
import numpy as np

from utils.dataset import normalize_log1p
from utils.pipeline import (
    SCORE_CLASSES_KEY,
    SCORE_KEY,
    create_argument_parser,
    run_annotation_pipeline,
    seed_python_numpy,
)


def train_celltypist(adata, seed, run_out_path, args):
    """Train CellTypist on log-normalized reference expression."""
    seed_python_numpy(seed)
    reference = normalize_log1p(adata)
    model = celltypist.train(
        X=reference,
        labels=args.train_celltype,
        check_expression=True,
    )
    if args.save_models:
        model_dir = run_out_path / "_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.write(str(model_dir / "model.pkl"))
    return model


def predict_celltypist(adata, model, pred_key, seed, run_out_path, args):
    """Annotate query cells and retain CellTypist class probabilities."""
    query = normalize_log1p(adata)
    result = celltypist.annotate(filename=query, model=model)

    predicted = result.predicted_labels.copy()
    predicted.index = predicted.index.astype(str)
    adata.obs[pred_key] = (
        predicted["predicted_labels"]
        .reindex(adata.obs_names.astype(str))
        .fillna("Unknown")
        .astype(str)
        .values
    )

    probabilities = result.probability_matrix.copy()
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
