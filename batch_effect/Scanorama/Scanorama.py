import os
import sys
import numpy as np
import scanpy as sc
import scanorama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline
from utils.dataset import preprocess_log_only


def integrate_scanorama(adata, seed, run_out_path, args):
    """Run Scanorama batch correction by splitting data by batch.

    Scanorama's published protocol expects log-normalized data on the full
    gene set and performs its own HVG selection internally, so we feed it
    via ``preprocess_log_only`` rather than the global HVG-2000 matrix.
    """
    adata_ls = [adata[adata.obs[args.batch_key] == batch, :].copy()
                for batch in np.unique(adata.obs[args.batch_key])]
    corrected = scanorama.correct_scanpy(
        adata_ls, seed=seed, return_dimred=True, hvg=2000
    )
    adata = sc.concat(corrected)
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scanorama')
    args = parser.parse_args()
    run_integration_pipeline("scanorama", integrate_scanorama, "X_scanorama", args,
                             preprocess_fn=preprocess_log_only)

