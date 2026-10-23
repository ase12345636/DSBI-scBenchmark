import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from biobatchnet import correct_batch_effects

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _copy_source_expression(adata):
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_biobatchnet(adata, label_col, batch_col):
    """Provide raw expression to BioBatchNet's native RNA preprocessor."""
    del label_col, batch_col
    return _copy_source_expression(adata)


def integrate_biobatchnet(adata, seed, run_out_path, args):
    """Run BioBatchNet without exposing cell-type labels to the model."""
    del run_out_path
    pl.seed_everything(seed, workers=True)
    bio_embedding, _ = correct_batch_effects(
        adata,
        batch_key=args.batch_key,
        cell_type_key=None,
        data_type="seq",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    adata.obsm["X_BioBatchNet"] = np.asarray(
        bio_embedding, dtype=np.float32
    )
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("BioBatchNet")
    args = parser.parse_args()
    run_integration_pipeline(
        "BioBatchNet",
        integrate_biobatchnet,
        "X_BioBatchNet",
        args,
        run_pca=False,
        preprocess_fn=preprocess_biobatchnet,
    )
