import os
import sys
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import (create_argument_parser, run_integration_pipeline,
                            export_adata_for_r, import_embedding_from_r)

R_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SMNN.R")
R_EXCHANGE_DIR_NAME = "r_exchange"


def _validate_source_counts(matrix):
    """Reject processed X before using it as SMNN's raw expression input."""
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
                'SMNN requires raw source expression in layers["counts"] or '
                "integer-valued X."
            )


def prepare_smnn_input(adata, label_col, batch_col):
    """Put full-gene source expression in X for SMNN's own workflow.

    Official SMNN accepts an unprocessed expression matrix.  Normalization for
    downstream feature selection and PCA remains local to ``SMNN.R``; no shared
    benchmark QC, normalization, or HVG filter is applied here.
    """
    del label_col, batch_col
    adata_smnn = adata.copy()
    if "counts" in adata_smnn.layers:
        adata_smnn.X = adata_smnn.layers["counts"].copy()
    else:
        _validate_source_counts(adata_smnn.X)
        adata_smnn.layers["counts"] = adata_smnn.X.copy()

    print(f"SMNN full-gene source-expression shape: {adata_smnn.shape}")
    return adata_smnn


def integrate_smnn(adata, seed, run_out_path, args):
    """Run SMNN 1.2.1 batch correction via an R subprocess.

    1. Export adata as mtx + csv once (shared across runs, skipped if exists)
    2. Call SMNN.R (supervised SMNN; seed changes each run for final PCA)
    3. Read back the 50-dimensional PCA embedding
    """
    adata.obs_names_make_unique()

    celltype_key = getattr(args, "celltype_key", None)
    if not celltype_key:
        raise ValueError(
            "SMNN is supervised. Please provide --celltype_key with the cell-type "
            "annotation column used by SMNN."
        )

    # Shared exchange dir at tool level — export once, but rebuild it if a
    # previous process stopped partway through the export.
    r_exchange_dir = os.path.join(args.save_path, "SMNN", R_EXCHANGE_DIR_NAME)
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

    # Force reticulate to use this exact Python environment, where mnnpy is installed.
    r_env = os.environ.copy()
    r_env["RETICULATE_PYTHON"] = sys.executable

    # Call R script (SMNN itself is deterministic; seed controls the final PCA)
    subprocess.run([
        "Rscript", R_SCRIPT_PATH,
        "--input_dir", r_exchange_dir,
        "--batch_key", args.batch_key,
        "--celltype_key", celltype_key,
        "--seed", str(seed),
    ], check=True, env=r_env)

    # Read back integrated embedding
    import_embedding_from_r(
        adata,
        os.path.join(r_exchange_dir, "embedding.csv"),
        "X_smnn",
    )

    # Read algorithm-only runtime from R (excludes file I/O)
    timing_path = os.path.join(r_exchange_dir, "timing.csv")
    algo_runtime = pd.read_csv(timing_path)["runtime_seconds"].iloc[0]

    return adata, algo_runtime


if __name__ == "__main__":
    parser = create_argument_parser("SMNN")

    # Add only when the shared parser does not already define it.
    if not any(action.dest == "celltype_key" for action in parser._actions):
        parser.add_argument(
            "--celltype_key",
            type=str,
            required=True,
            help="Cell-type annotation column used by supervised SMNN",
        )

    args = parser.parse_args()
    r_exchange_dir = os.path.join(args.save_path, "SMNN", R_EXCHANGE_DIR_NAME)
    run_integration_pipeline(
        "SMNN", integrate_smnn, "X_smnn", args,
        r_exchange_dir=r_exchange_dir,
        preprocess_fn=prepare_smnn_input,
        run_pca=False,
    )
