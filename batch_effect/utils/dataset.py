"""Shared cell and gene quality control for integration pipelines."""

import scanpy as sc
import scipy.sparse as sp


_SPARSE_QC_LENGTH_ERROR = (
    "Length of values (1) does not match length of index"
)


def _calculate_qc_metrics_compat(adata):
    """Run Scanpy QC with a narrow compatibility path for Scanpy < 1.8.1.

    Older Scanpy passes the two-dimensional ``numpy.matrix`` returned by a
    sparse row sum directly to pandas.  With the scDML environment's pandas,
    the resulting Series has an invalid outer length of one.  Scanpy 1.8.1
    fixed this by flattening those row sums.

    Keep the normal Scanpy path for every environment where it works.  Only
    retry the identified legacy failure with ``csr_array``, whose row sums are
    one-dimensional, and restore the original sparse-matrix object before any
    method-specific preprocessing runs.
    """
    kwargs = {
        "qc_vars": ["mt"],
        "percent_top": None,
        "log1p": False,
        "inplace": True,
    }

    try:
        sc.pp.calculate_qc_metrics(adata, **kwargs)
        return
    except ValueError as exc:
        is_legacy_sparse_shape_error = (
            sp.issparse(adata.X)
            and _SPARSE_QC_LENGTH_ERROR in str(exc)
        )
        if not is_legacy_sparse_shape_error:
            raise

    original_x = adata.X
    try:
        adata.X = sp.csr_array(original_x)
        sc.pp.calculate_qc_metrics(adata, **kwargs)
    finally:
        adata.X = original_x


def quality_control(adata, min_genes=200, min_cells=3, mt_threshold=5.0):
    """Apply the original scSCI/scQuartz QC before method-specific preprocessing."""
    result = adata.copy()
    if "counts" in result.layers:
        result.X = result.layers["counts"].copy()

    sc.pp.filter_cells(result, min_genes=min_genes)
    sc.pp.filter_genes(result, min_cells=min_cells)
    result.var["mt"] = result.var_names.str.upper().str.startswith("MT-")
    _calculate_qc_metrics_compat(result)
    return result[result.obs["pct_counts_mt"] < mt_threshold].copy()
