import os
import sys

import concord as ccd
import scanpy as sc
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _copy_source_expression(adata):
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_concord(adata, label_col, batch_col):
    """Apply the preprocessing in the official CONCORD quick start."""
    del label_col, batch_col
    adata = _copy_source_expression(adata)

    features = ccd.ul.select_features(
        adata,
        n_top_features=min(2_000, adata.n_vars),
        flavor="seurat_v3",
    )
    adata.var["highly_variable"] = adata.var_names.isin(features)
    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)
    adata.raw = adata
    return adata


def integrate_concord(adata, seed, run_out_path, args):
    """Run unsupervised CONCORD with its documented defaults."""
    del run_out_path
    model = ccd.Concord(
        adata=adata,
        input_feature=adata.var_names[
            adata.var["highly_variable"]
        ].tolist(),
        domain_key=args.batch_key,
        seed=seed,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        save_dir=None,
    )
    model.fit_transform(
        output_key="X_CONCORD",
        return_class=False,
        return_class_prob=False,
        save_model=False,
    )
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("CONCORD")
    args = parser.parse_args()
    run_integration_pipeline(
        "CONCORD",
        integrate_concord,
        "X_CONCORD",
        args,
        run_pca=False,
        preprocess_fn=preprocess_concord,
    )
