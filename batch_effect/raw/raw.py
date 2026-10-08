import os
import sys
import numpy as np
import scanpy as sc
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def preprocess_raw_baseline(
    adata,
    label_col,
    batch_col,
    min_genes=200,
    min_cells=3,
    mt_threshold=5.0,
    target_sum=1e4,
    n_top_genes=2000,
):
    """Apply the benchmark's intentionally fixed unintegrated-PCA design."""
    ad_raw = adata.copy()
    if "counts" in ad_raw.layers:
        ad_raw.X = ad_raw.layers["counts"].copy()
    else:
        values = (
            ad_raw.X.data
            if sp.issparse(ad_raw.X)
            else np.asarray(ad_raw.X).reshape(-1)
        )
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or not np.all(
                np.isclose(values, np.rint(values), rtol=0.0, atol=1e-6)
            )
        ):
            raise ValueError(
                'Raw baseline requires source expression in layers["counts"] '
                "or integer-valued X."
            )
        ad_raw.layers["counts"] = ad_raw.X.copy()

    sc.pp.filter_cells(ad_raw, min_genes=min_genes)
    sc.pp.filter_genes(ad_raw, min_cells=min_cells)
    ad_raw.var["mt"] = ad_raw.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        ad_raw, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    ad_raw = ad_raw[ad_raw.obs["pct_counts_mt"] < mt_threshold].copy()
    sc.pp.normalize_total(ad_raw, target_sum=target_sum)
    sc.pp.log1p(ad_raw)
    if ad_raw.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            ad_raw,
            n_top_genes=n_top_genes,
            flavor="seurat",
            batch_key=batch_col,
        )
        ad_raw = ad_raw[:, ad_raw.var["highly_variable"]].copy()
    print(f"Filtered shape (raw baseline): {ad_raw.shape}")
    return ad_raw


def integrate_raw(adata, seed, run_out_path, args):
    """Raw baseline: only compute PCA (no batch correction)."""
    sc.tl.pca(adata, random_state=seed)
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('raw')
    args = parser.parse_args()
    run_integration_pipeline(
        "raw",
        integrate_raw,
        "X_pca",
        args,
        run_pca=False,
        preprocess_fn=preprocess_raw_baseline,
    )
