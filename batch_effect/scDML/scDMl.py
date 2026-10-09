import os
import sys
import numpy as np
import scipy.sparse as sp
from scDML import scDMLModel

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


def prepare_scdml_input(adata, label_col, batch_col):
    """Put source expression in ``X``; scDML performs all other preprocessing."""
    del label_col, batch_col  # Required by the shared preprocessing callback API.

    adata_scdml = adata.copy()
    if "counts" in adata_scdml.layers:
        adata_scdml.X = adata_scdml.layers["counts"].copy()
    else:
        _validate_raw_x_fallback(adata_scdml.X)
        adata_scdml.layers["counts"] = adata_scdml.X.copy()

    print(f"Source-expression shape for scDML: {adata_scdml.shape}")
    return adata_scdml


def _unused_obs_key(adata, stem):
    """Return a temporary obs key that cannot overwrite input metadata."""
    key = stem
    suffix = 1
    while key in adata.obs:
        key = f"{stem}_{suffix}"
        suffix += 1
    return key


def integrate_scdml(adata, seed, run_out_path, args):
    """Run scDML with its native preprocessing and integration defaults."""
    del run_out_path

    # scDML's Normalization helper reserves an obs column literally named
    # ``batch``.  A temporary alias prevents it from overwriting datasets whose
    # user-supplied batch key is also ``batch``.
    batch_key = _unused_obs_key(adata, "_scdml_input_batch")
    obs_name_key = _unused_obs_key(adata, "_scdml_original_obs_name")
    adata.obs[batch_key] = adata.obs[args.batch_key].copy()
    adata.obs[obs_name_key] = adata.obs_names.astype(str)

    model = scDMLModel()

    # Leave every algorithmic argument at the installed scDML defaults:
    # 1,000 HVGs, per-batch z-score scaling, 100 PCs, and Louvain resolution 3.
    adata_scdml = model.preprocess(adata, batch_key=batch_key)

    # AnnData.concatenate(), used inside scDML for per-batch scaling, appends
    # suffixes to cell IDs.  Restore stable IDs so pre/post benchmark metrics
    # can align the same cells.
    adata_scdml.obs_names = adata_scdml.obs[obs_name_key].astype(str).to_numpy()
    adata_scdml.obs[args.batch_key] = adata_scdml.obs[batch_key].copy()
    del adata_scdml.obs[obs_name_key]

    model.integrate(
        adata_scdml,
        batch_key=batch_key,
        ncluster_list=[15],
        expect_num_cluster=15,
        seed=seed,
    )
    del adata_scdml.obs[batch_key]
    return adata_scdml


if __name__ == '__main__':
    parser = create_argument_parser('scDML')
    args = parser.parse_args()
    run_integration_pipeline(
        "scDML",
        integrate_scdml,
        "X_emb",
        args,
        run_pca=False,
        preprocess_fn=prepare_scdml_input,
    )
