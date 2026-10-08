import os
import sys
import numpy as np
import scanpy as sc
import scanorama
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _validate_raw_x_fallback(matrix):
    """Reject a processed ``X`` before treating it as source expression."""
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
                'No layers["counts"] was found, and adata.X is not finite, '
                "nonnegative integer source expression. Provide source "
                'expression in layers["counts"].'
            )


def preprocess_scanorama(adata, label_col, batch_col, target_sum=1e4):
    """Prepare the full-gene input from the official Scanorama protocol."""
    del label_col, batch_col  # Required by the shared preprocessing callback API.

    adata_scanorama = adata.copy()
    if "counts" in adata_scanorama.layers:
        adata_scanorama.X = adata_scanorama.layers["counts"].copy()
    else:
        _validate_raw_x_fallback(adata_scanorama.X)
        adata_scanorama.layers["counts"] = adata_scanorama.X.copy()

    sc.pp.normalize_total(adata_scanorama, target_sum=target_sum)
    sc.pp.log1p(adata_scanorama)

    print(
        "Scanorama input shape (log-normalized, all genes): "
        f"{adata_scanorama.shape}"
    )
    return adata_scanorama


def integrate_scanorama(adata, seed, run_out_path, args):
    """Run Scanorama integration by splitting data by batch.

    Scanorama receives log-normalized data on the full gene set.  ``hvg`` is
    deliberately omitted so ``integrate_scanpy`` retains its API default
    (``hvg=None``) instead of applying benchmark-specific feature selection.
    The integration API adds ``X_scanorama`` without unnecessarily correcting
    the full expression matrix, matching the official Scanorama protocol.
    """
    # Preserve first-appearance batch order, as in the official protocol.
    adata_ls = [adata[adata.obs[args.batch_key] == batch, :].copy()
                for batch in adata.obs[args.batch_key].unique()]
    scanorama.integrate_scanpy(adata_ls, seed=seed)
    adata = sc.concat(adata_ls)
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scanorama')
    args = parser.parse_args()
    run_integration_pipeline("scanorama", integrate_scanorama, "X_scanorama", args,
                             run_pca=False,
                             preprocess_fn=preprocess_scanorama)
