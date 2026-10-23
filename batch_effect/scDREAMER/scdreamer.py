import os
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow.compat.v1 as tf

SCDREAMER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCDREAMER_DIR)
sys.path.insert(0, os.path.dirname(SCDREAMER_DIR))
from src.model import scDREAMER
from utils.pipeline import create_argument_parser, run_integration_pipeline


SCDREAMER_PROFILES = {
    "immune": (150, 1, 0),
    "lung": (250, 2, 1),
    "pancreas": (300, 1, 0),
}


def resolve_scdreamer_profile(dataset_path, requested_profile, matrix):
    profile = requested_profile
    if profile is None:
        dataset_name = os.path.basename(dataset_path).lower()
        profile = next(
            (name for name in SCDREAMER_PROFILES if name in dataset_name),
            None,
        )

    if profile is not None:
        epochs, shuffle_type, sparse_input = SCDREAMER_PROFILES[profile]
        return profile, epochs, shuffle_type, sparse_input

    # The paper has no dataset-specific profile for other inputs. Use the
    # official constructor defaults and select only the matching input reader.
    return "official_default", 300, 1, int(sparse.issparse(matrix))


def prepare_scdreamer_input(adata, label_col, batch_col):
    """Give scDREAMER source expression; its official code preprocesses it."""
    del label_col, batch_col
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    return adata


def integrate_scdreamer(adata, seed, run_out_path, args):
    """Run unsupervised scDREAMER without exposing cell-type labels."""
    random.seed(seed)
    np.random.seed(seed)
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    input_path = os.path.join(run_out_path, f"scdreamer_input_{seed}.h5ad")
    model_input = ad.AnnData(
        X=adata.X.copy(),
        obs=pd.DataFrame(
            {
                "batch": np.asarray(adata.obs[args.batch_key].astype(str)),
                "_dummy_label": "unknown",
            },
            index=adata.obs_names.copy(),
        ),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    model_input.write_h5ad(input_path)

    profile, epochs, shuffle_type, sparse_input = resolve_scdreamer_profile(
        args.dataset_path,
        getattr(args, "scdreamer_profile", None),
        adata.X,
    )
    print(
        f"scDREAMER profile: {profile} "
        f"(epochs={epochs}, shuffle_type={shuffle_type}, sparseIP={sparse_input})"
    )

    prefix = os.path.join(run_out_path, "scDREAMER_")
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.333
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as session:
        model = scDREAMER(
            session,
            batch="batch",
            cell_type="_dummy_label",
            name=prefix,
            epoch=epochs,
            X_dim=min(2000, adata.n_vars),
            dataset_name=input_path,
            shuffle_type=shuffle_type,
            sparseIP=sparse_input,
        )
        model.train_cluster()

    embedding = np.loadtxt(
        f"{prefix}latent_matrix_{epochs - 1}.csv",
        delimiter=",",
    )
    if embedding.ndim == 1:
        embedding = embedding.reshape(adata.n_obs, -1)
    if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs:
        raise ValueError(
            "scDREAMER returned an embedding that does not match the input cells."
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("scDREAMER returned a non-finite embedding.")

    adata.obsm["X_scDREAMER"] = embedding.astype(np.float32, copy=False)
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("scDREAMER")
    parser.add_argument(
        "--scdreamer_profile",
        choices=tuple(SCDREAMER_PROFILES),
        default=None,
        help="Override automatic Immune/Lung/Pancreas paper settings.",
    )
    args = parser.parse_args()
    run_integration_pipeline(
        "scDREAMER",
        integrate_scdreamer,
        "X_scDREAMER",
        args,
        run_pca=False,
        preprocess_fn=prepare_scdreamer_input,
    )
