import os
import sys
import subprocess
import numpy as np
import scanpy as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _copy_source_expression(adata):
    """Put the source expression matrix in X without imposing count checks."""
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_sccm(adata, label_col, batch_col):
    """Apply the input preprocessing described by the scCM paper."""
    del label_col, batch_col  # scCM selects HVGs globally, not per batch.
    adata = _copy_source_expression(adata)


    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(3_000, adata.n_vars),
        flavor="seurat",
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


def integrate_sccm(adata, seed, run_out_path, args):
    sccm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ScCM")
    input_path = os.path.join(run_out_path, f"sccm_input_{seed}.h5ad")
    output_dir = os.path.join(run_out_path, f"sccm_{seed}")

    adata.write_h5ad(input_path)

    subprocess.run([
        sys.executable, os.path.join(sccm_dir, "scCM.py"),
        "--input_h5ad_path", input_path,
        "--obs_label_colname", args.celltype_key,
        # Defaults reported in the scCM paper (Methods), which calls these the
        # default training settings.  They take precedence over the conflicting
        # larger-batch example in the repository README.
        "--workers", "1",
        "--epochs", "10",
        "--start_epoch", "0",
        "--batch_size", str(min(512, adata.n_obs)),
        "--lr", "1e-6",
        "--wd", "1e-6",
        "--moco-dim", "128",
        "--moco-m", "0.99",
        "--moco-t", "0.8",
        "--warmup-epochs", "5",
        "--seed", str(seed),
        "--gpu", "0",
        "--save_dir", output_dir,
    ], check=True)

    embedding_path = os.path.join(
        output_dir, "SCCM", f"feature_scCM_sccm_input_{seed}.csv"
    )
    adata.obsm["X_scCM"] = np.loadtxt(
        embedding_path, delimiter=","
    ).astype(np.float32)

    return adata


if __name__ == "__main__":
    parser = create_argument_parser("scCM")
    args = parser.parse_args()
    run_integration_pipeline(
        "scCM",
        integrate_sccm,
        "X_scCM",
        args,
        run_pca=False,
        preprocess_fn=preprocess_sccm,
    )
