import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

from scalex import SCALEX

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def prepare_scalex_input(adata, label_col, batch_col):
    """Give SCALEX source expression; its official code preprocesses it."""
    del label_col, batch_col
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    return adata


def integrate_scalex(adata, seed, run_out_path, args):
    """Run the vendored official SCALEX model with its published defaults."""
    model_input = ad.AnnData(
        X=adata.X.copy(),
        obs=pd.DataFrame(
            {"batch": adata.obs[args.batch_key].astype(str).to_numpy()},
            index=adata.obs_names.copy(),
        ),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    model_result = SCALEX(
        model_input,
        batch_name="batch",
        min_features=0,
        seed=seed,
        gpu=0,
        outdir=run_out_path,
        ignore_umap=True,
        show=False,
    )
    embedding = np.asarray(model_result.obsm["latent"], dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[0] != model_result.n_obs:
        raise ValueError(
            "SCALEX returned an embedding that does not match its output cells."
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("SCALEX returned a non-finite embedding.")

    result = adata[model_result.obs_names].copy()
    result.obsm["X_SCALEX"] = embedding
    return result


if __name__ == "__main__":
    parser = create_argument_parser("SCALEX")
    args = parser.parse_args()
    run_integration_pipeline(
        "SCALEX",
        integrate_scalex,
        "X_SCALEX",
        args,
        run_pca=False,
        preprocess_fn=prepare_scalex_input,
    )
