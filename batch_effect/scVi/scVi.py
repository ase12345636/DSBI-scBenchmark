import os
import sys
import numpy as np
import scanpy as sc
import scvi
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _prepare_count_input(adata):
    """Put the experiment-designated raw-count source in ``X``.

    Existing ``layers["counts"]`` values are authoritative even when
    non-integer. In particular, the immune Villani batch must be retained.
    """
    ad_scvi = adata.copy()
    if "counts" in ad_scvi.layers:
        ad_scvi.X = ad_scvi.layers["counts"].copy()
        return ad_scvi

    values = (
        ad_scvi.X.data
        if sp.issparse(ad_scvi.X)
        else np.asarray(ad_scvi.X).reshape(-1)
    )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.all(np.isclose(values, np.rint(values), rtol=0.0, atol=1e-6))
    ):
        raise ValueError(
            'scVI requires source counts in layers["counts"] or integer-valued X.'
        )
    ad_scvi.layers["counts"] = ad_scvi.X.copy()
    return ad_scvi


def preprocess_scvi(
    adata,
    label_col,
    batch_col,
    target_sum=1e4,
    n_top_genes=2000,
):
    """Prepare input following the official scvi-tools preprocessing tutorial."""
    ad_scvi = _prepare_count_input(adata)
    sc.pp.normalize_total(ad_scvi, target_sum=target_sum)
    sc.pp.log1p(ad_scvi)
    ad_scvi.raw = ad_scvi
    if ad_scvi.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            ad_scvi, n_top_genes=n_top_genes, flavor="seurat_v3",
            layer="counts", batch_key=batch_col, subset=True,
        )
    print(f"Filtered shape (scVI counts-HVG): {ad_scvi.shape}")
    return ad_scvi


def integrate_scvi(adata, seed, run_out_path, args):
    """Run scVI batch correction.

    scVI models raw integer counts with a NB / ZINB likelihood, so we point
    setup_anndata at ``layers["counts"]`` (kept by preprocess_scvi)
    instead of the log-normalized ``X``.
    """
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=args.batch_key)
    # Hyperparameters per official scvi-tools harmonization tutorial:
    # https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html
    model = scvi.model.SCVI(
        adata, n_layers=2, n_latent=30, gene_likelihood="nb"
    )
    model.train()
    adata.obsm["X_scVI"] = model.get_latent_representation()
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scVi')
    args = parser.parse_args()
    run_integration_pipeline(
        "scVi", integrate_scvi, "X_scVI", args,
        preprocess_fn=preprocess_scvi,
        run_pca=False,
    )
