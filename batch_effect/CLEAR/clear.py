import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def prepare_clear_input(adata, label_col, batch_col):
    """Apply CLEAR's published batch-effect preprocessing after shared QC."""
    del label_col, batch_col
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    # CLEAR's batch-effect reproduction uses --norm --scale, without the
    # optional log or HVG flags. Shared QC already supplies its cell/gene
    # filtering step for the common benchmark cohort.
    sc.pp.normalize_total(
        adata,
        target_sum=10_000,
        exclude_highly_expressed=True,
    )
    adata.raw = adata
    sc.pp.scale(adata, max_value=10, zero_center=True)
    return adata


def integrate_clear(adata, seed, run_out_path, args):
    """Run the unchanged official CLEAR entry point without GT labels."""
    del args
    input_path = os.path.join(run_out_path, f"clear_input_{seed}.h5ad")
    model_matrix = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    model_input = ad.AnnData(
        X=np.asarray(model_matrix, dtype=np.float32),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    model_input.write_h5ad(input_path)

    official_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "CLEAR",
    )
    subprocess.run(
        [
            sys.executable,
            os.path.join(official_dir, "CLEAR.py"),
            "--input_h5ad_path", input_path,
            "--workers", "1",
            # Settings from the official batch-effect reproduction command.
            "--epochs", "500",
            "--batch_size", "512",
            "--lr", "1",
            "--wd", "1e-6",
            "--pcl_r", "1024",
            "--moco_m", "0.999",
            "--temperature", "0.2",
            "--cos",
            "--seed", str(seed),
            "--gpu", "0",
            "--save_dir", run_out_path,
        ],
        check=True,
    )

    embedding_path = os.path.join(
        run_out_path,
        "CLEAR",
        f"feature_CLEAR_clear_input_{seed}.csv",
    )
    embedding = np.loadtxt(embedding_path, delimiter=",")
    if embedding.ndim == 1:
        embedding = embedding.reshape(adata.n_obs, -1)
    if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs:
        raise ValueError(
            "CLEAR returned an embedding that does not match the input cells."
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("CLEAR returned a non-finite embedding.")

    adata.obsm["X_CLEAR"] = embedding.astype(np.float32, copy=False)
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("CLEAR")
    args = parser.parse_args()
    run_integration_pipeline(
        "CLEAR",
        integrate_clear,
        "X_CLEAR",
        args,
        run_pca=False,
        preprocess_fn=prepare_clear_input,
    )
