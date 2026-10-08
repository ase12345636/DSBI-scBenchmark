import os
import sys
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import harmonypy as hm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def preprocess_harmony(
    adata,
    label_col,
    batch_col,
    target_sum=1e4,
    n_top_genes=2000,
    n_comps=20,
):
    """Prepare scaled PCs following Harmony's official Seurat workflow."""
    ad_harmony = adata.copy()
    if "counts" in ad_harmony.layers:
        ad_harmony.X = ad_harmony.layers["counts"].copy()
    else:
        values = (
            ad_harmony.X.data
            if sp.issparse(ad_harmony.X)
            else np.asarray(ad_harmony.X).reshape(-1)
        )
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or not np.all(
                np.isclose(values, np.rint(values), rtol=0.0, atol=1e-6)
            )
        ):
            raise ValueError(
                'Harmony preprocessing requires source expression in '
                'layers["counts"] or integer-valued X.'
            )
        ad_harmony.layers["counts"] = ad_harmony.X.copy()

    sc.pp.normalize_total(ad_harmony, target_sum=target_sum)
    sc.pp.log1p(ad_harmony)
    if ad_harmony.n_vars > n_top_genes:
        # The official Harmony Seurat vignette selects 2,000 variable features
        # within each condition and runs PCA on their union.
        hvg_union = set()
        for batch_value in ad_harmony.obs[batch_col].unique():
            ad_batch = ad_harmony[
                ad_harmony.obs[batch_col] == batch_value
            ].copy()
            sc.pp.highly_variable_genes(
                ad_batch,
                n_top_genes=n_top_genes,
                flavor="seurat_v3",
                layer="counts",
            )
            hvg_union.update(
                ad_batch.var_names[ad_batch.var["highly_variable"]].tolist()
            )
        ad_harmony = ad_harmony[
            :, ad_harmony.var_names.isin(hvg_union)
        ].copy()
    sc.pp.scale(ad_harmony, max_value=10)
    sc.tl.pca(
        ad_harmony,
        n_comps=min(n_comps, ad_harmony.n_obs - 1, ad_harmony.n_vars - 1),
        random_state=0,
    )
    print(
        f"Harmony input: {ad_harmony.shape}, "
        f"{ad_harmony.obsm['X_pca'].shape[1]} PCs"
    )
    return ad_harmony


def integrate_harmony(adata, seed, run_out_path, args):
    """Run Harmony batch correction on PCA embeddings."""
    del run_out_path
    harmony_result = hm.run_harmony(
        adata.obsm["X_pca"],
        adata.obs,
        args.batch_key,
        random_state=seed,
    )
    corrected = np.asarray(harmony_result.Z_corr)
    if corrected.shape[0] == adata.n_obs:
        adata.obsm["X_pca_harmony"] = corrected
    elif corrected.shape[1] == adata.n_obs:
        adata.obsm["X_pca_harmony"] = corrected.T
    else:
        raise ValueError(
            "harmonypy returned an embedding that cannot be aligned to cells: "
            f"{corrected.shape} for {adata.n_obs} cells"
        )
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('harmony')
    args = parser.parse_args()
    run_integration_pipeline(
        "harmony",
        integrate_harmony,
        "X_pca_harmony",
        args,
        run_pca=False,
        preprocess_fn=preprocess_harmony,
    )
