import os
import random
import sys
import types
from types import SimpleNamespace

import keras
import keras.backend as K
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
from scipy.sparse import issparse

# Keep the vendored SCIDRL source unchanged while providing the two legacy
# Keras APIs it expects under the GPU-capable TensorFlow/Keras runtime.
keras_engine = types.ModuleType("keras.engine")
keras_engine.Layer = keras.layers.Layer
sys.modules.setdefault("keras.engine", keras_engine)
keras.optimizers.Adam = keras.optimizers.legacy.Adam

from scidrl.main import SCIDRL_train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _copy_source_expression(adata):
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_scidrl(adata, label_col, batch_col):
    """Apply the Seurat-v3-style preprocessing used in the SCIDRL paper."""
    del label_col
    adata = _copy_source_expression(adata)

    hvg_names = set()
    for batch in adata.obs[batch_col].dropna().unique():
        batch_adata = adata[adata.obs[batch_col] == batch].copy()
        sc.pp.highly_variable_genes(
            batch_adata,
            n_top_genes=min(1_000, batch_adata.n_vars),
            flavor="seurat_v3",
        )
        hvg_names.update(
            batch_adata.var_names[batch_adata.var["highly_variable"]]
        )

    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var_names.isin(hvg_names)].copy()
    return adata


def integrate_scidrl(adata, seed, run_out_path, args):
    data_path = os.path.join(run_out_path, "scidrl_expression.csv")
    metadata_path = os.path.join(run_out_path, "scidrl_metadata.csv")

    expression = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    pd.DataFrame(
        expression,
        index=adata.obs_names,
        columns=adata.var_names,
    ).to_csv(data_path)
    pd.DataFrame(
        {"batch": adata.obs[args.batch_key].astype(str).to_numpy()},
        index=adata.obs_names,
    ).to_csv(metadata_path)

    K.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    n_batches = adata.obs[args.batch_key].nunique()
    parameters = SimpleNamespace(
        zdim=16,
        znoise_dim=2,
        epochs=500,
        batch_size=100,
        lr=1e-3,
        gamma=1,
        fg_lambda=1 if n_batches == 2 else n_batches,
        acts="sigmoid" if n_batches == 2 else "softmax",
        minmaxscale=True,
    )

    model = SCIDRL_train(parameters, data_path, metadata_path)
    model.train()
    embedding, _ = model.infer()

    output_names = pd.Index(model.meta.index.astype(str))
    order = output_names.get_indexer(adata.obs_names.astype(str))
    if np.any(order < 0):
        raise ValueError("SCIDRL output is missing input cells")
    adata.obsm["X_SCIDRL"] = embedding[
        order, parameters.znoise_dim:
    ].astype(np.float32)
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("SCIDRL")
    args = parser.parse_args()
    run_integration_pipeline(
        "SCIDRL",
        integrate_scidrl,
        "X_SCIDRL",
        args,
        run_pca=False,
        preprocess_fn=preprocess_scidrl,
    )
