import os
import sys

import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from scipy.spatial import cKDTree

try:
    from mnnpy.utils import (
        adjust_shift_variance,
        compute_correction,
        transform_input_data,
    )
except ImportError as exc:
    raise ImportError(
        "SMNN requires mnnpy. Install it in the active environment with "
        "`pip install mnnpy`."
    ) from exc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _ordered_unique(values):
    """Return unique values in first-observation order."""
    return list(dict.fromkeys(values))


def _resolve_batch_order(batch_labels, reference_batch=None):
    """Choose a deterministic correction order, optionally moving a reference first."""
    batch_order = _ordered_unique(batch_labels.tolist())
    if len(batch_order) < 2:
        raise ValueError("SMNN requires at least two batches")

    if reference_batch is None:
        return batch_order

    matches = [batch for batch in batch_order if str(batch) == reference_batch]
    if not matches:
        available = ", ".join(map(str, batch_order))
        raise ValueError(
            f"SMNN reference batch '{reference_batch}' was not found. "
            f"Available batches: {available}"
        )

    reference = matches[0]
    return [reference] + [batch for batch in batch_order if batch != reference]


def _query_neighbors(reference, query, k, n_jobs):
    """Query a KD-tree and always return a two-dimensional index array."""
    indices = cKDTree(reference).query(query, k=k, workers=n_jobs)[1]
    return np.asarray(indices, dtype=np.int64).reshape(query.shape[0], k)


def _find_mutual_nn(data1, data2, k, n_jobs):
    """Find MNN pairs while safely handling cell types smaller than ``k``."""
    if data1.shape[0] == 0 or data2.shape[0] == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    k1 = min(k, data1.shape[0])
    k2 = min(k, data2.shape[0])
    neighbors_1 = _query_neighbors(data1, data2, k1, n_jobs)
    neighbors_2 = _query_neighbors(data2, data1, k2, n_jobs)

    mutual_1 = []
    mutual_2 = []
    for index_2, candidates_1 in enumerate(neighbors_1):
        for index_1 in candidates_1:
            if np.any(neighbors_2[index_1] == index_2):
                mutual_1.append(index_1)
                mutual_2.append(index_2)

    return (
        np.asarray(mutual_1, dtype=np.int32),
        np.asarray(mutual_2, dtype=np.int32),
    )


def _find_supervised_pairs(ref_data, new_data, ref_labels, new_labels, k, n_jobs):
    """Find MNNs separately inside every cell type shared by two batches."""
    shared_labels = [
        label
        for label in _ordered_unique(ref_labels.tolist())
        if np.any(new_labels == label)
    ]
    if not shared_labels:
        raise ValueError(
            "SMNN found no shared cell-type labels between the reference and "
            "current batch"
        )

    all_ref_pairs = []
    all_new_pairs = []
    for label in shared_labels:
        ref_indices = np.flatnonzero(ref_labels == label)
        new_indices = np.flatnonzero(new_labels == label)
        local_ref, local_new = _find_mutual_nn(
            np.ascontiguousarray(ref_data[ref_indices], dtype=np.float32),
            np.ascontiguousarray(new_data[new_indices], dtype=np.float32),
            k=k,
            n_jobs=n_jobs,
        )
        if local_ref.size == 0:
            continue

        all_ref_pairs.append(ref_indices[local_ref])
        all_new_pairs.append(new_indices[local_new])
        print(
            f"  Cell type {label!r}: {ref_indices.size} reference cells, "
            f"{new_indices.size} current cells, {local_ref.size} MNN pairs"
        )

    if not all_ref_pairs:
        raise RuntimeError("SMNN did not find any mutual nearest-neighbor pairs")

    return (
        np.ascontiguousarray(np.concatenate(all_ref_pairs), dtype=np.int32),
        np.ascontiguousarray(np.concatenate(all_new_pairs), dtype=np.int32),
        shared_labels,
    )


def _as_batch_matrices(adata, batch_labels, batch_order):
    """Split X by batch without changing the original observation order."""
    batch_indices = [np.flatnonzero(batch_labels == batch) for batch in batch_order]
    matrices = []
    for indices in batch_indices:
        matrix = adata.X[indices]
        if issparse(matrix):
            matrix = matrix.tocsr()
        matrices.append(matrix)
    return matrices, batch_indices


def supervised_mnn_correct(
    matrices,
    label_vectors,
    var_names,
    k=20,
    sigma=1.0,
    var_adj=True,
    n_jobs=1,
):
    """Apply the SMNN 1.2.1 correction procedure to cell-by-gene matrices.

    This follows the reference implementation's defaults: cosine normalization
    on input and output, MNN detection restricted to matched cell types, a
    Gaussian-smoothed correction vector, and optional variance adjustment.
    Unlike the upstream R loop, reference labels are expanded after every batch,
    so supervised matching remains valid for more than two batches.
    """
    if len(matrices) < 2:
        raise ValueError("SMNN requires at least two batch matrices")
    if len(matrices) != len(label_vectors):
        raise ValueError("Each SMNN batch matrix must have one label vector")
    n_features = matrices[0].shape[1]
    if n_features < 3:
        raise ValueError(
            "SMNN needs at least three retained features; check the preprocessing "
            "and highly-variable-gene filters"
        )
    for batch_id, (matrix, labels) in enumerate(zip(matrices, label_vectors)):
        if matrix.shape[1] != n_features:
            raise ValueError("All SMNN batch matrices must contain the same features")
        if matrix.shape[0] != len(labels):
            raise ValueError(
                f"SMNN batch {batch_id} has {matrix.shape[0]} cells but "
                f"{len(labels)} cell-type labels"
            )
    if k < 1:
        raise ValueError("SMNN k must be at least 1")
    if sigma <= 0:
        raise ValueError("SMNN sigma must be greater than 0")

    print("Performing SMNN cosine normalization...")
    in_batches, out_batches, _, same_set = transform_input_data(
        matrices,
        cos_norm_in=True,
        cos_norm_out=True,
        var_index=list(map(str, var_names)),
        var_subset=None,
        n_jobs=n_jobs,
    )
    if not same_set:
        raise RuntimeError("Unexpected SMNN input/output feature mismatch")

    corrected_batches = [None] * len(matrices)
    corrected_batches[0] = np.ascontiguousarray(out_batches[0], dtype=np.float32)
    ref_data = np.ascontiguousarray(in_batches[0], dtype=np.float32)
    ref_labels = np.asarray(label_vectors[0], dtype=str)
    run_stats = []

    for target in range(1, len(matrices)):
        print(f"SMNN correction step {target}/{len(matrices) - 1}")
        new_data = np.ascontiguousarray(in_batches[target], dtype=np.float32)
        new_labels = np.asarray(label_vectors[target], dtype=str)
        mnn_ref, mnn_new, shared_labels = _find_supervised_pairs(
            ref_data,
            new_data,
            ref_labels,
            new_labels,
            k=k,
            n_jobs=n_jobs,
        )

        print(f"  Computing correction vectors from {mnn_ref.size} MNN pairs...")
        correction = compute_correction(
            ref_data,
            new_data,
            mnn_ref,
            mnn_new,
            new_data,
            float(sigma),
        )
        correction = np.ascontiguousarray(correction, dtype=np.float32)

        if var_adj:
            print("  Adjusting correction-vector variance...")
            correction = adjust_shift_variance(
                ref_data,
                new_data,
                correction,
                float(sigma),
                n_jobs,
            )

        new_data = np.ascontiguousarray(new_data + correction, dtype=np.float32)
        corrected_batches[target] = new_data
        ref_data = np.ascontiguousarray(
            np.concatenate((ref_data, new_data), axis=0), dtype=np.float32
        )
        ref_labels = np.concatenate((ref_labels, new_labels))
        run_stats.append(
            {
                "target_index": target,
                "n_mnn_pairs": int(mnn_ref.size),
                "shared_celltypes": [str(label) for label in shared_labels],
            }
        )

    return corrected_batches, run_stats


def integrate_smnn(adata, seed, run_out_path, args):
    """Run supervised MNN correction and expose a PCA embedding for evaluation."""
    required_columns = [args.batch_key, args.celltype_key]
    missing_columns = [key for key in required_columns if key not in adata.obs]
    if missing_columns:
        raise KeyError(f"Missing AnnData obs columns: {missing_columns}")
    if adata.obs[required_columns].isna().any().any():
        raise ValueError("SMNN batch and cell-type labels cannot contain missing values")

    batch_labels = adata.obs[args.batch_key].to_numpy()
    batch_order = _resolve_batch_order(batch_labels, args.smnn_reference_batch)
    matrices, batch_indices = _as_batch_matrices(adata, batch_labels, batch_order)
    label_vectors = [
        adata.obs.iloc[indices][args.celltype_key].astype(str).to_numpy()
        for indices in batch_indices
    ]

    available_cpus = os.cpu_count() or 1
    n_jobs = available_cpus if args.smnn_n_jobs < 1 else args.smnn_n_jobs
    n_jobs = max(1, min(n_jobs, available_cpus))

    print(f"SMNN batch order: {[str(batch) for batch in batch_order]}")
    print(
        f"SMNN supervision key: {args.celltype_key}; k={args.smnn_k}; "
        f"sigma={args.smnn_sigma}; var_adj={not args.smnn_no_var_adj}; "
        f"n_jobs={n_jobs}"
    )
    corrected_batches, run_stats = supervised_mnn_correct(
        matrices,
        label_vectors,
        adata.var_names,
        k=args.smnn_k,
        sigma=args.smnn_sigma,
        var_adj=not args.smnn_no_var_adj,
        n_jobs=n_jobs,
    )

    corrected = np.empty(adata.shape, dtype=np.float32)
    for indices, matrix in zip(batch_indices, corrected_batches):
        corrected[indices] = matrix

    n_comps = min(50, corrected.shape[0] - 1, corrected.shape[1] - 1)
    if n_comps < 2:
        raise ValueError("SMNN needs at least three cells and three retained features")
    adata.obsm["X_smnn"] = sc.pp.pca(
        corrected,
        n_comps=n_comps,
        random_state=seed,
        zero_center=True,
    )
    adata.uns["smnn"] = {
        "supervised": True,
        "supervision_key": args.celltype_key,
        "batch_order": [str(batch) for batch in batch_order],
        "k": args.smnn_k,
        "sigma": args.smnn_sigma,
        "variance_adjustment": not args.smnn_no_var_adj,
        "n_jobs": n_jobs,
        "steps": run_stats,
    }
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("SMNN")
    parser.add_argument(
        "--smnn_k",
        type=int,
        default=20,
        help="Number of nearest neighbors used by SMNN (official default: 20)",
    )
    parser.add_argument(
        "--smnn_sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing bandwidth (official default: 1)",
    )
    parser.add_argument(
        "--smnn_n_jobs",
        type=int,
        default=-1,
        help="Parallel workers; values below 1 use all available CPUs",
    )
    parser.add_argument(
        "--smnn_reference_batch",
        type=str,
        default=None,
        help="Reference batch value; default is the first observed batch",
    )
    parser.add_argument(
        "--smnn_no_var_adj",
        action="store_true",
        help="Disable the official variance-adjustment step",
    )
    args = parser.parse_args()
    run_integration_pipeline(
        "SMNN",
        integrate_smnn,
        "X_smnn",
        args,
    )
