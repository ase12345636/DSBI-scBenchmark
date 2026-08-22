"""SingleR reference-based cell type annotation benchmark."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline import (
    create_argument_parser,
    predict_r_reference,
    run_annotation_pipeline,
    train_r_reference,
)

R_SCRIPT = Path(__file__).with_suffix(".R")


def train_singler(adata, seed, run_out_path, args):
    return train_r_reference(adata, seed, run_out_path, args, R_SCRIPT)


def predict_singler(adata, model_path, pred_key, seed, run_out_path, args):
    return predict_r_reference(
        adata,
        model_path,
        pred_key,
        seed,
        run_out_path,
        args,
        R_SCRIPT,
        score_type="continuous_scores",
    )


if __name__ == "__main__":
    parser = create_argument_parser("SingleR")
    arguments = parser.parse_args()
    run_annotation_pipeline(
        "SingleR", "singler_prediction", train_singler, predict_singler, arguments
    )
