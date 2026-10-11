"""Shared cell and gene quality control for integration pipelines."""

import scanpy as sc


def quality_control(adata, min_genes=200, min_cells=3, mt_threshold=5.0):
    """Apply the original scSCI/scQuartz QC before method-specific preprocessing."""
    result = adata.copy()
    if "counts" in result.layers:
        result.X = result.layers["counts"].copy()

    sc.pp.filter_cells(result, min_genes=min_genes)
    sc.pp.filter_genes(result, min_cells=min_cells)
    result.var["mt"] = result.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        result, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True,
    )
    return result[result.obs["pct_counts_mt"] < mt_threshold].copy()
