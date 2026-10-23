import os
import sys

import numpy as np
import scanpy as sc
import scvi
from fadvi import FADVI
from fadvi._fadvi import SemiSupervisedTrainingPlanFixed
from scvi.train import TrainingPlan

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


class _UnsupervisedTrainingPlan(SemiSupervisedTrainingPlanFixed):
    """Skip meaningless one-class label metrics in fully unsupervised mode."""

    def compute_and_log_metrics(self, loss_output, metrics, mode):
        TrainingPlan.compute_and_log_metrics(self, loss_output, metrics, mode)


class UnsupervisedFADVI(FADVI):
    _training_plan_cls = _UnsupervisedTrainingPlan


def _copy_source_expression(adata):
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_fadvi(adata, label_col, batch_col):
    """Select count-based HVGs while retaining raw counts for FADVI."""
    del label_col
    adata = _copy_source_expression(adata)
    if adata.n_vars > 2_000:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=2_000,
            flavor="seurat_v3",
            layer="counts",
            batch_key=batch_col,
            subset=True,
        )
    return adata


def integrate_fadvi(adata, seed, run_out_path, args):
    """Run the official fully-unsupervised FADVI configuration."""
    del run_out_path
    scvi.settings.seed = seed

    dummy_key = "_fadvi_dummy_label"
    adata.obs[dummy_key] = "dummy"
    UnsupervisedFADVI.setup_anndata(
        adata,
        batch_key=args.batch_key,
        labels_key=dummy_key,
        unlabeled_category="Unknown",
        layer="counts",
    )

    # FADVI's current metric logger asks torchmetrics to score a one-class
    # dummy label, which torchmetrics rejects.  Reusing scvi's base ELBO logger
    # changes logging only; the model, losses, and optimization stay official.
    model = UnsupervisedFADVI(
        adata,
        n_latent_l=30,
        n_latent_b=30,
        n_layers=2,
        lambda_b=50,
        lambda_l=0,
    )
    model.train(max_epochs=30, batch_size=200)
    adata.obsm["X_FADVI"] = np.asarray(
        model.get_latent_representation(representation="lr"),
        dtype=np.float32,
    )
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("FADVI")
    args = parser.parse_args()
    run_integration_pipeline(
        "FADVI",
        integrate_fadvi,
        "X_FADVI",
        args,
        run_pca=False,
        preprocess_fn=preprocess_fadvi,
    )
