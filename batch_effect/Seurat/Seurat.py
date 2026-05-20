import os
import sys
import subprocess
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import (create_argument_parser, run_integration_pipeline,
                            export_adata_for_r, import_embedding_from_r)
from utils.dataset import preprocess_filter_only

R_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Seurat.R")
R_EXCHANGE_DIR_NAME = "r_exchange"


def integrate_seurat(adata, seed, run_out_path, args):
    """Run Seurat v5 batch correction via R subprocess.

    1. Export adata as mtx + csv once (shared across runs, skipped if exists)
    2. Call Seurat.R (RPCA integration, seed changes each run)
    3. Read back embedding CSV
    """
    adata.obs_names_make_unique()

    # Shared exchange dir at tool level — export only on first run
    r_exchange_dir = os.path.join(args.save_path, "Seurat", R_EXCHANGE_DIR_NAME)
    if not os.path.exists(os.path.join(r_exchange_dir, "matrix.mtx")):
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
        preprocess_fn=preprocess_filter_only,
        run_pca=False,
    )
