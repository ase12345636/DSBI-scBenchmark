import os
import sys
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import (create_argument_parser, run_integration_pipeline,
                            export_adata_for_r, import_embedding_from_r)

R_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Seurat.R")
R_EXCHANGE_DIR_NAME = "r_exchange"


def _validate_source_counts(matrix):
    """Reject processed X before exporting it as Seurat's count matrix."""
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).reshape(-1)
    chunk_size = 1_000_000
    for start in range(0, values.size, chunk_size):
        chunk = values[start:start + chunk_size]
        if (
            not np.all(np.isfinite(chunk))
            or np.any(chunk < 0)
            or not np.all(
                np.isclose(chunk, np.rint(chunk), rtol=0.0, atol=1e-6)
            )
        ):
            raise ValueError(
                'Seurat requires source counts in layers["counts"] or '
                "integer-valued X."
            )


def prepare_seurat_input(adata, label_col, batch_col):
    """Put the source count matrix in X for Seurat's own preprocessing.

    Seurat performs normalization and feature selection in ``Seurat.R``.  The
    Python side therefore does not apply the benchmark-wide QC, normalization,
    or HVG filters.
    """
    del label_col, batch_col
    adata_seurat = adata.copy()
    if "counts" in adata_seurat.layers:
        adata_seurat.X = adata_seurat.layers["counts"].copy()
    else:
        _validate_source_counts(adata_seurat.X)
        adata_seurat.layers["counts"] = adata_seurat.X.copy()

    print(f"Seurat source-count shape: {adata_seurat.shape}")
    return adata_seurat


def integrate_seurat(adata, seed, run_out_path, args):
    """Run Seurat v5 batch correction via R subprocess.

    1. Export adata as mtx + csv once (shared across runs, skipped if exists)
    2. Call Seurat.R (default CCA integration, seed changes each run)
    3. Read back embedding CSV
    """
    adata.obs_names_make_unique()

    # Shared exchange dir at tool level — export only on first run
    r_exchange_dir = os.path.join(args.save_path, "Seurat", R_EXCHANGE_DIR_NAME)
    exchange_inputs = (
        "matrix.mtx",
        "barcodes.csv",
        "features.csv",
        "metadata.csv",
    )
    if not all(
        os.path.isfile(os.path.join(r_exchange_dir, name))
        and os.path.getsize(os.path.join(r_exchange_dir, name)) > 0
        for name in exchange_inputs
    ):
        export_adata_for_r(adata, r_exchange_dir)

    # Call R script (seed differs each run)
    subprocess.run([
        "Rscript", R_SCRIPT_PATH,
        "--input_dir", r_exchange_dir,
        "--batch_key", args.batch_key,
        "--seed", str(seed),
    ], check=True)

    # Read back integrated embedding
    import_embedding_from_r(adata, os.path.join(r_exchange_dir, "embedding.csv"), "X_seurat")

    # Read algorithm-only runtime from R (excludes I/O overhead)
    timing_path = os.path.join(r_exchange_dir, "timing.csv")
    algo_runtime = pd.read_csv(timing_path)['runtime_seconds'].iloc[0]

    return adata, algo_runtime


if __name__ == '__main__':
    parser = create_argument_parser('Seurat')
    args = parser.parse_args()
    r_exchange_dir = os.path.join(args.save_path, "Seurat", R_EXCHANGE_DIR_NAME)
    run_integration_pipeline(
        "Seurat", integrate_seurat, "X_seurat", args,
        r_exchange_dir=r_exchange_dir,
        preprocess_fn=prepare_seurat_input,
        run_pca=False,
    )
