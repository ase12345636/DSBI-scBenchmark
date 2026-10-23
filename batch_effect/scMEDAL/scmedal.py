import os
import sys

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import train_test_split

from scMEDAL.scMEDAL import DomainAdversarialAE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def preprocess_scmedal(adata, label_col, batch_col):
    """Apply scMEDAL's normalization, log transform, and HVG selection."""
    del label_col, batch_col
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    if adata.n_vars > 3_000:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=3_000,
            flavor="seurat",
            subset=True,
        )
    print(f"scMEDAL input shape (log-normalized HVGs): {adata.shape}")
    return adata


def _minmax_dense(matrix):
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    minimum = float(matrix.min())
    maximum = float(matrix.max())
    if maximum <= minimum:
        raise ValueError("scMEDAL input has no expression variation")
    return (matrix - minimum) / (maximum - minimum)


def integrate_scmedal(adata, seed, run_out_path, args):
    """Train the label-free fixed-effects scMEDAL network."""
    del run_out_path
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    batch_codes = (
        adata.obs[args.batch_key]
        .astype("category")
        .cat.codes.to_numpy(dtype=np.int32)
    )
    if np.any(batch_codes < 0):
        raise ValueError(f"{args.batch_key} contains missing values")
    n_batches = int(batch_codes.max()) + 1
    batch_one_hot = tf.keras.utils.to_categorical(
        batch_codes,
        num_classes=n_batches,
    ).astype(np.float32)

    train_indices, validation_indices = train_test_split(
        np.arange(adata.n_obs),
        test_size=0.2,
        random_state=seed,
        stratify=batch_codes,
    )
    train_expression = _minmax_dense(adata.X[train_indices])
    validation_expression = _minmax_dense(adata.X[validation_indices])

    model = DomainAdversarialAE(
        in_shape=train_expression.shape,
        n_clusters=n_batches,
        n_latent_dims=50,
        layer_units=[512, 132],
        last_activation="linear",
        get_pred=False,
        use_batch_norm=True,
    )
    optimizer_class = tf.keras.optimizers.legacy.Adam
    model.compile(
        loss_recon=tf.keras.losses.MeanSquaredError(),
        loss_multiclass=tf.keras.losses.CategoricalCrossentropy(),
        metric_multiclass=tf.keras.metrics.CategoricalAccuracy(name="acc"),
        opt_autoencoder=optimizer_class(learning_rate=1e-4),
        opt_adversary=optimizer_class(learning_rate=1e-4),
        loss_recon_weight=4_000,
        loss_gen_weight=1,
        loss_class_weight=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_recon_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        (train_expression, batch_one_hot[train_indices]),
        train_expression,
        validation_data=(
            (
                validation_expression,
                batch_one_hot[validation_indices],
            ),
            validation_expression,
        ),
        epochs=500,
        batch_size=min(512, train_indices.size),
        callbacks=[early_stopping],
        verbose=2,
    )

    train_outputs = model.encoder.predict(
        train_expression,
        batch_size=min(512, train_indices.size),
        verbose=0,
    )
    validation_outputs = model.encoder.predict(
        validation_expression,
        batch_size=min(512, validation_indices.size),
        verbose=0,
    )
    train_embedding = np.asarray(train_outputs[-1], dtype=np.float32)
    validation_embedding = np.asarray(
        validation_outputs[-1], dtype=np.float32
    )
    embedding = np.empty(
        (adata.n_obs, train_embedding.shape[1]), dtype=np.float32
    )
    embedding[train_indices] = train_embedding
    embedding[validation_indices] = validation_embedding
    adata.obsm["X_scMEDAL"] = embedding
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("scMEDAL")
    args = parser.parse_args()
    run_integration_pipeline(
        "scMEDAL",
        integrate_scmedal,
        "X_scMEDAL",
        args,
        run_pca=False,
        preprocess_fn=preprocess_scmedal,
    )
