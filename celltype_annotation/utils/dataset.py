"""Dataset utilities shared by cell type annotation benchmarks."""

from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy import sparse


def load_dataset(path):
    """Load an AnnData dataset and make cell/gene identifiers unique."""
    print(f"Loading data from {path}")
    adata = sc.read_h5ad(path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def is_count_matrix(matrix, sample_size=10000, tolerance=1e-6):
    """Return True when sampled non-zero values look like raw integer counts."""
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        return False
    if values.size > sample_size:
        indices = np.linspace(0, values.size - 1, sample_size, dtype=int)
        values = values[indices]
    values = np.asarray(values, dtype=float)
    return bool(
        np.all(np.isfinite(values))
        and np.min(values) >= 0
        and np.all(np.abs(values - np.rint(values)) <= tolerance)
    )


def normalize_log1p(adata, target_sum=1e4, *, copy=True):
    """Normalize raw counts to 10,000 counts/cell and log1p-transform.

    Matrices that already contain non-integer values are treated as processed
    expression and returned unchanged. Set copy=False only when the caller
    already owns a disposable AnnData copy.
    """
    out = adata.copy() if copy else adata
    if is_count_matrix(out.X):
        sc.pp.normalize_total(out, target_sum=target_sum)
        sc.pp.log1p(out)
    return out


def prepare_umap(adata, seed, n_top_genes=2000):
    """Compute a test-expression UMAP without dropping or reordering cells."""
    # This runs after prediction/evaluation on a disposable per-run AnnData.
    # Reusing it avoids another full copy of a potentially very large query.
    vis = normalize_log1p(adata, copy=False)

    if vis.n_obs < 3:
        raise ValueError("At least three test cells are required to compute UMAP")

    nonzero = np.asarray((vis.X != 0).sum(axis=0)).ravel() > 0
    if not np.all(nonzero):
        vis = vis[:, nonzero].copy()
    if vis.n_vars < 2:
        raise ValueError("At least two expressed genes are required to compute UMAP")

    if vis.n_vars > n_top_genes:
        try:
            sc.pp.highly_variable_genes(
                vis, n_top_genes=n_top_genes, flavor="seurat", subset=True
            )
        except Exception as exc:
            print(f"[UMAP] HVG selection failed; using all genes: {exc}")

    n_comps = min(50, vis.n_obs - 1, vis.n_vars - 1)
    sc.tl.pca(vis, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(
        vis,
        use_rep="X_pca",
        n_neighbors=min(15, vis.n_obs - 1),
        random_state=seed,
    )
    sc.tl.umap(vis, random_state=seed)
    return vis
