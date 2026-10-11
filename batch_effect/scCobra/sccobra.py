import os
import sys

import numpy as np
import scanpy as sc
from scCobra import scCobra
from sklearn.preprocessing import MaxAbsScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _copy_source_expression(adata):
    """Put the source expression matrix in X without imposing count checks."""
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_sccobra(adata, label_col, batch_col):
    """Apply scCobra's native RNA preprocessing defaults.

    Keeping this deterministic work here excludes preprocessing from the
    measured model runtime. scCobra's ``processed=True`` loading path applies
    MaxAbsScaler once more; that operation is idempotent after this native
    scaling and therefore does not change the model input.
    """
    del label_col  # Labels are retained in obs but are not used for preprocessing.
    adata = _copy_source_expression(adata)

    # This matches scCobra.preprocessing_rna: batches with ten or fewer cells
    # cannot form a useful training batch and are removed before other filters.
    batch_counts = adata.obs[batch_col].value_counts()
    valid_batches = batch_counts[batch_counts > 10].index
    adata = adata[adata.obs[batch_col].isin(valid_batches)].copy()
    if adata.n_obs == 0:
        raise ValueError("scCobra preprocessing removed every batch (<=10 cells each).")

    # Native RNA defaults remove mitochondrial/ERCC features rather than using
    # a mitochondrial-percentage cell filter.
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(2_000, adata.n_vars),
        flavor="seurat",
        batch_key=batch_col,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    scaler = MaxAbsScaler()
    adata.X = scaler.fit_transform(adata.X)
    adata.var["maxabs_max"] = scaler.max_abs_.astype(np.float32)
    adata.var["maxabs_min"] = 0

    print(
        "scCobra preprocessing: "
        f"{adata.n_obs} cells, {adata.n_vars} genes "
        "(native RNA defaults including MaxAbs scaling)"
    )
    return adata


def integrate_sccobra(adata, seed, run_out_path, args):
    """Run scCobra batch correction."""
    adata = scCobra(
        adata, batch_name=args.batch_key, processed=True,
        outdir=run_out_path, show=False, gpu=0,
        ignore_umap=True, seed=seed
    )
    return adata


if __name__ == '__main__':
    parser = create_argument_parser('scCobra')
    args = parser.parse_args()
    run_integration_pipeline(
        "scCobra",
        integrate_sccobra,
        "latent",
        args,
        run_pca=False,
        preprocess_fn=preprocess_sccobra,
    )
