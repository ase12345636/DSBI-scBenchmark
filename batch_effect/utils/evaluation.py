import heapq
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2, entropy
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_samples,
)
from sklearn.neighbors import NearestNeighbors


SCIB_METRIC_KEYS = [
    "NMI_cluster/label",
    "ARI_cluster/label",
    "ASW_label",
    "ASW_label/batch",
    "BRAS",
    "PCR_batch",
    "cell_cycle_conservation",
    "isolated_label_F1",
    "isolated_label_silhouette",
    "graph_conn",
    "kBET",
    "iLISI",
    "CiLISI",
    "cLISI",
    "hvg_overlap",
    "trajectory",
    "morans_i",
]

class RootCellError(Exception):
    pass


class NeighborsError(Exception):
    pass


def _check_adata(adata_obj):
    if type(adata_obj) is not ad.AnnData:
        raise TypeError("Input is not a valid AnnData object")


def _check_batch(batch, obs):
    if batch not in obs:
        raise ValueError(f"column {batch} is not in obs")


def _ensure_category(adata_obj, key):
    if not pd.api.types.is_categorical_dtype(adata_obj.obs[key]):
        adata_obj.obs[key] = adata_obj.obs[key].astype("category")


def _safe_metric(metric_name, fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception as exc:
        print(f"[metric:{metric_name}] failed: {exc}")
        return np.nan


def _align_pre_post(adata_pre, adata_post):
    common = adata_pre.obs_names.intersection(adata_post.obs_names)
    if len(common) == 0:
        raise ValueError("No common cells between pre and post AnnData")
    return adata_pre[common].copy(), adata_post[common].copy()


# ============================================================
# Basic helpers matching scIB behavior
# ============================================================

def get_resolutions(n=20, min_val=0, max_val=2):
    res_range = max_val - min_val
    return [res_range * (x + 1) / n for x in range(n)]


def nmi_score(
    adata,
    cluster_key,
    label_key,
    implementation="arithmetic",
):
    _check_adata(adata)
    _check_batch(cluster_key, adata.obs)
    _check_batch(label_key, adata.obs)

    clusters = adata.obs[cluster_key].tolist()
    labels = adata.obs[label_key].tolist()

    if implementation not in ["max", "min", "geometric", "arithmetic"]:
        raise ValueError(
            "Only sklearn NMI implementations are supported in local evaluator"
        )

    return normalized_mutual_info_score(
        clusters,
        labels,
        average_method=implementation,
    )


def ari_score(adata, cluster_key, label_key):
    _check_adata(adata)
    _check_batch(cluster_key, adata.obs)
    _check_batch(label_key, adata.obs)
    return adjusted_rand_score(
        adata.obs[cluster_key].to_numpy(),
        adata.obs[label_key].to_numpy(),
    )


def cluster_optimal_resolution(
    adata,
    label_key,
    cluster_key,
    cluster_function=None,
    metric=None,
    resolutions=None,
    use_rep=None,
    force=False,
    verbose=True,
    return_all=False,
    metric_kwargs=None,
    **kwargs,
):
    if cluster_function is None:
        cluster_function = sc.tl.leiden

    if metric is None:
        metric = nmi_score

    if metric_kwargs is None:
        metric_kwargs = {}

    if resolutions is None:
        resolutions = get_resolutions(n=10, max_val=2)

    score_max = 0.0
    res_max = resolutions[0]
    clustering = None
    score_all = []

    for res in resolutions:
        resolution_key = f"{cluster_key}_{res}"

        if resolution_key not in adata.obs.columns or force:
            knn_rep = adata.uns.get("neighbors", {}).get("params", {}).get("use_rep")
            if use_rep is not None and use_rep != knn_rep:
                sc.pp.neighbors(adata, use_rep=use_rep)

            cluster_function(adata, resolution=res, key_added=resolution_key, **kwargs)

        score = metric(
            adata,
            cluster_key=resolution_key,
            label_key=label_key,
            **metric_kwargs,
        )
        score_all.append(score)

        if verbose:
            print(f"resolution: {res}, score: {score}")

        if score_max < score:
            score_max = score
            res_max = res
            clustering = adata.obs[resolution_key]

    adata.obs[cluster_key] = clustering
    score_df = pd.DataFrame(zip(resolutions, score_all), columns=["resolution", "score"])

    if return_all:
        return res_max, score_max, score_df
    return res_max, score_max


def _label_silhouette_samples(
    adata,
    label_key,
    embed,
    metric="euclidean",
    cache=None,
):
    cache_key = (id(adata), label_key, embed, metric)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    values = silhouette_samples(
        adata.obsm[embed],
        adata.obs[label_key],
        metric=metric,
    )
    if cache is not None:
        cache[cache_key] = values
    return values


def silhouette_label(
    adata,
    label_key,
    embed,
    metric="euclidean",
    scale=True,
    silhouette_cache=None,
):
    if embed not in adata.obsm.keys():
        raise KeyError(f"{embed} not in obsm")

    asw = np.mean(
        _label_silhouette_samples(
            adata,
            label_key=label_key,
            embed=embed,
            metric=metric,
            cache=silhouette_cache,
        )
    )
    if scale:
        asw = (asw + 1.0) / 2.0
    return asw


def silhouette_batch(
    adata,
    batch_key,
    label_key,
    embed,
    metric="euclidean",
    return_all=False,
    scale=True,
    verbose=True,
):
    if embed not in adata.obsm.keys():
        raise KeyError(f"{embed} not in obsm")

    sil_per_label = []
    for group in adata.obs[label_key].unique():
        adata_group = adata[adata.obs[label_key] == group]
        n_batches = adata_group.obs[batch_key].nunique()

        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue

        sil = silhouette_samples(
            adata_group.obsm[embed],
            adata_group.obs[batch_key],
            metric=metric,
        )

        sil = [abs(i) for i in sil]
        if scale:
            sil = [1.0 - i for i in sil]

        sil_per_label.extend([(group, score) for score in sil])

    sil_df = pd.DataFrame.from_records(sil_per_label, columns=["group", "silhouette_score"])

    if len(sil_per_label) == 0:
        sil_means = np.nan
        asw = np.nan
    else:
        sil_means = sil_df.groupby("group").mean()
        asw = sil_means["silhouette_score"].mean()

    if verbose:
        print(f"mean silhouette per group: {sil_means}")

    if return_all:
        return asw, sil_means, sil_df
    return asw


def bras(adata, batch_key, label_key, embed):
    """Batch-removal-adapted silhouette using cosine mean-other distances."""
    if embed not in adata.obsm.keys():
        raise KeyError(f"{embed} not in obsm")

    scores = []
    for group in adata.obs[label_key].unique():
        mask = np.asarray(adata.obs[label_key] == group)
        embedding = adata.obsm[embed][mask]
        if issparse(embedding):
            embedding = embedding.toarray()
        embedding = np.asarray(embedding, dtype=float)

        # Re-factorize within this cell type. AnnData may retain unused global
        # categorical levels after subsetting, whose non-contiguous codes (for
        # example 0 and 8) cannot index a two-row per-group accumulator.
        batch_codes, batch_levels = pd.factorize(
            adata.obs.loc[mask, batch_key],
            sort=False,
        )
        if np.any(batch_codes < 0):
            raise ValueError(f"{batch_key} contains missing values")
        n_batches = len(batch_levels)
        n_cells = embedding.shape[0]
        if (n_batches == 1) or (n_batches == n_cells):
            continue

        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        normalized = np.divide(
            embedding,
            norms,
            out=np.zeros_like(embedding),
            where=norms > 0,
        )

        batch_counts = np.bincount(batch_codes)
        batch_sums = np.zeros((n_batches, normalized.shape[1]), dtype=float)
        np.add.at(batch_sums, batch_codes, normalized)

        own_counts = batch_counts[batch_codes]
        other_counts = n_cells - own_counts
        valid = (own_counts > 1) & (other_counts > 0)

        intra = np.zeros(n_cells, dtype=float)
        inter = np.zeros(n_cells, dtype=float)
        intra[valid] = 1.0 - np.einsum(
            "ij,ij->i",
            normalized[valid],
            batch_sums[batch_codes[valid]] - normalized[valid],
        ) / (own_counts[valid] - 1)
        inter[valid] = 1.0 - np.einsum(
            "ij,ij->i",
            normalized[valid],
            normalized.sum(axis=0) - batch_sums[batch_codes[valid]],
        ) / other_counts[valid]

        # Cosine distances are bounded by [0, 2]. Round-off in the
        # mean-other shortcut can otherwise make the silhouette unstable.
        np.clip(intra, 0.0, 2.0, out=intra)
        np.clip(inter, 0.0, 2.0, out=inter)
        intra[intra <= 1e-12] = 0.0
        inter[inter <= 1e-12] = 0.0

        denominator = np.maximum(intra[valid], inter[valid])
        silhouettes = np.zeros(n_cells, dtype=float)
        silhouettes[valid] = np.divide(
            inter[valid] - intra[valid],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > np.finfo(float).eps,
        )
        np.clip(silhouettes, -1.0, 1.0, out=silhouettes)
        scores.append(float(np.clip(np.mean(1.0 - np.abs(silhouettes)), 0.0, 1.0)))

    return float(np.mean(scores)) if scores else np.nan


def get_isolated_labels(adata, label_key, batch_key, iso_threshold, verbose):
    tmp = adata.obs[[label_key, batch_key]].drop_duplicates()
    batch_per_lab = tmp.groupby(label_key).agg({batch_key: "count"})

    if iso_threshold is None:
        iso_threshold = batch_per_lab.min().tolist()[0]

    if iso_threshold == adata.obs[batch_key].nunique():
        warnings.warn(
            "iso_threshold is equal to number of batches in data, no isolated labels will be found",
            stacklevel=2,
        )

    iso = batch_per_lab[batch_per_lab[batch_key] <= iso_threshold].index.tolist()
    if verbose:
        print(f"isolated labels: {iso}")
    return iso


def score_isolated_label(
    adata,
    label_key,
    isolated_label,
    embed,
    cluster_key,
    cluster=True,
    resolutions=None,
    scale=True,
    verbose=False,
    copy_adata=True,
):
    adata_local = adata.copy() if copy_adata else adata

    if cluster:
        def max_f1(adata_tmp, cluster_key, label_key, label, argmax=False):
            obs = adata_tmp.obs
            max_cluster = None
            max_score = 0.0
            for cluster_id in obs[cluster_key].unique():
                y_pred = obs[cluster_key] == cluster_id
                y_true = obs[label_key] == label
                f1 = f1_score(y_pred, y_true)
                if f1 > max_score:
                    max_score = f1
                    max_cluster = cluster_id
            if argmax:
                return max_cluster
            return max_score

        cluster_optimal_resolution(
            adata_local,
            label_key=label_key,
            cluster_key=cluster_key,
            use_rep=embed,
            metric=max_f1,
            metric_kwargs={"label": isolated_label},
            resolutions=resolutions,
            force=False,
            verbose=verbose,
        )

        score = max_f1(
            adata_local,
            cluster_key=cluster_key,
            label_key=label_key,
            label=isolated_label,
            argmax=False,
        )
    else:
        if "silhouette_temp" not in adata_local.obs:
            adata_local.obs["silhouette_temp"] = silhouette_samples(
                adata_local.obsm[embed],
                adata_local.obs[label_key],
            )

        score = adata_local.obs[
            adata_local.obs[label_key] == isolated_label
        ]["silhouette_temp"].mean()

        if scale:
            score = (score + 1.0) / 2.0

    if verbose:
        print(f"{isolated_label}: {score}")
    return score


def isolated_labels_metric(
    adata,
    label_key,
    batch_key,
    embed,
    cluster=True,
    cluster_key="iso_label",
    resolutions=None,
    iso_threshold=None,
    scale=True,
    return_all=False,
    verbose=True,
    silhouette_cache=None,
):
    isolated = get_isolated_labels(
        adata,
        label_key=label_key,
        batch_key=batch_key,
        iso_threshold=iso_threshold,
        verbose=verbose,
    )

    scores = {}
    adata_local = adata.copy()
    if not cluster:
        adata_local.obs["silhouette_temp"] = _label_silhouette_samples(
            adata,
            label_key=label_key,
            embed=embed,
            metric="euclidean",
            cache=silhouette_cache,
        )

    for label in isolated:
        scores[label] = score_isolated_label(
            adata_local,
            label_key=label_key,
            isolated_label=label,
            embed=embed,
            cluster_key=cluster_key,
            cluster=cluster,
            resolutions=resolutions,
            scale=scale,
            verbose=verbose,
            copy_adata=False,
        )

    scores = pd.Series(scores)
    if return_all:
        return scores
    return scores.mean() if len(scores) > 0 else np.nan


def graph_connectivity(adata, label_key):
    if "neighbors" not in adata.uns:
        raise KeyError("Please compute the neighborhood graph before running graph_connectivity")

    _ensure_category(adata, label_key)
    clust_res = []

    for label in adata.obs[label_key].cat.categories:
        adata_sub = adata[adata.obs[label_key].isin([label])]
        _, labels = connected_components(
            adata_sub.obsp["connectivities"],
            connection="strong",
        )
        tab = pd.Series(labels).value_counts()
        clust_res.append(tab.max() / tab.sum())

    return float(np.mean(clust_res)) if clust_res else np.nan


# ============================================================
# PCR metrics
# ============================================================

def _to_covariate_2d(covariate):
    if isinstance(covariate, pd.Series):
        return covariate.to_numpy(copy=False)[:, None]
    if isinstance(covariate, pd.DataFrame):
        return covariate.to_numpy(copy=False)

    covariate_arr = np.asarray(covariate)
    if covariate_arr.ndim == 1:
        covariate_arr = covariate_arr[:, None]
    return covariate_arr


def _encode_covariate(covariate):
    covariate_arr = _to_covariate_2d(covariate)
    cov_df = pd.DataFrame(covariate_arr)

    is_numeric = cov_df.dtypes.map(pd.api.types.is_numeric_dtype).all()
    if is_numeric:
        return cov_df.to_numpy(dtype=float)

    encoded = pd.get_dummies(cov_df, drop_first=True)
    if encoded.shape[1] == 0:
        return np.zeros((cov_df.shape[0], 1), dtype=float)
    return encoded.to_numpy(dtype=float)


def _linreg_r2_multi(X, Y):
    n_samples = X.shape[0]
    X_design = np.concatenate([np.ones((n_samples, 1), dtype=float), X], axis=1)

    coef, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    y_pred = X_design @ coef

    ss_res = np.sum((Y - y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
    r2[~np.isfinite(r2)] = 0.0
    return r2


def pc_regression(
    data,
    covariate,
    pca_var=None,
    n_comps=50,
    svd_solver="arpack",
):
    if isinstance(data, (np.ndarray, sp_sparse.csr_matrix, sp_sparse.csc_matrix)):
        matrix = data
    else:
        raise TypeError("invalid type: data must be numpy array or sparse matrix")

    if pca_var is None:
        if n_comps is None or n_comps > min(matrix.shape):
            n_comps = min(matrix.shape)

        if n_comps == min(matrix.shape):
            svd_solver = "full"
            if sp_sparse.issparse(matrix):
                matrix = matrix.toarray()

        X_pca, _, _, pca_var = sc.tl.pca(
            matrix,
            n_comps=n_comps,
            use_highly_variable=False,
            return_info=True,
            svd_solver=svd_solver,
            copy=True,
        )
    else:
        X_pca = matrix

    cov_X = _encode_covariate(covariate)
    if cov_X.shape[1] == 0:
        return 0.0

    if sp_sparse.issparse(X_pca):
        X_pca = X_pca.toarray()
    X_pca = np.asarray(X_pca, dtype=float)

    if X_pca.ndim == 1:
        X_pca = X_pca[:, None]

    r2 = _linreg_r2_multi(cov_X, X_pca)
    r2 = np.maximum(0.0, r2)

    pca_var = np.asarray(pca_var, dtype=float)
    var_weights = pca_var / np.sum(pca_var)
    return float(np.sum(r2 * var_weights))


def pcr(
    adata,
    covariate,
    embed=None,
    n_comps=50,
    recompute_pca=False,
):
    _check_adata(adata)
    _check_batch(covariate, adata.obs)

    cov_values = adata.obs[covariate]

    if embed is not None:
        if embed not in adata.obsm:
            raise KeyError(f"{embed} not found in adata.obsm")
        return pc_regression(
            adata.obsm[embed],
            cov_values,
            n_comps=n_comps,
        )

    if (not recompute_pca) and ("X_pca" in adata.obsm) and ("pca" in adata.uns):
        return pc_regression(
            adata.obsm["X_pca"],
            cov_values,
            pca_var=adata.uns["pca"]["variance"],
        )

    return pc_regression(
        adata.X,
        cov_values,
        n_comps=n_comps,
    )


def pcr_comparison(
    adata_pre,
    adata_post,
    covariate,
    embed=None,
    n_comps=50,
    recompute_pca=False,
    scale=True,
):
    if embed == "X_pca":
        embed = None

    pcr_before = pcr(
        adata_pre,
        covariate=covariate,
        recompute_pca=recompute_pca,
        n_comps=n_comps,
    )

    pcr_after = pcr(
        adata_post,
        covariate=covariate,
        embed=embed,
        recompute_pca=recompute_pca,
        n_comps=n_comps,
    )

    if scale:
        if pcr_before <= 0:
            return 0.0
        score = (pcr_before - pcr_after) / pcr_before
        if score < 0:
            score = 0.0
        return float(score)

    return float(pcr_after - pcr_before)


# ============================================================
# Cell cycle conservation
# ============================================================

def _resource_organism_name(organism):
    organism_map = {
        "mouse": "mus_musculus",
        "mus musculus": "mus_musculus",
        "mus_musculus": "mus_musculus",
        "human": "homo_sapiens",
        "homo sapiens": "homo_sapiens",
        "homo_sapiens": "homo_sapiens",
    }
    key = organism.lower()
    if key not in organism_map:
        raise ValueError(
            f"organism '{organism}' not supported. supported: {list(organism_map.keys())}"
        )
    return organism_map[key]


def get_cell_cycle_genes(organism="mouse"):
    org_name = _resource_organism_name(organism)
    gene_file = (
        Path(__file__).resolve().parent
        / "resources"
        / f"cell_cycle_genes_{org_name}.tsv"
    )
    if not gene_file.exists():
        raise FileNotFoundError(
            f"Cell-cycle resource file not found: {gene_file}. "
            "Please add scIB-compatible resource TSV files."
        )
    return pd.read_table(gene_file)


def _restore_scquartz_log_expression(adata_obj):
    """Undo scQuartz's final z-scaling for expression-based metrics only."""
    required = {"mean", "std", "scquartz_expression_zero"}
    if not required.issubset(adata_obj.var.columns):
        return adata_obj

    matrix = adata_obj.X
    if issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix).copy()
    dtype = matrix.dtype
    mean = adata_obj.var["mean"].to_numpy(dtype=dtype, copy=False)
    std = adata_obj.var["std"].to_numpy(dtype=dtype, copy=False)
    matrix *= std
    matrix += mean
    np.maximum(matrix, 0.0, out=matrix)
    adata_obj.X = matrix
    return adata_obj


def infer_cell_cycle_organism(adata_obj):
    """Infer human versus mouse from scIB marker overlap with ``var_names``."""
    var_names = set(map(str, adata_obj.var_names))
    overlaps = {}
    for organism in ("human", "mouse"):
        gene_map = get_cell_cycle_genes(organism)
        overlaps[organism] = max(
            sum(str(gene) in var_names for gene in gene_map[column].dropna())
            for column in ("gene_name", "gene_id")
        )

    organism = max(overlaps, key=overlaps.get)
    if overlaps[organism] == 0:
        raise ValueError(
            "cell cycle genes not found in adata.var_names for human or mouse"
        )
    return organism


def score_cell_cycle(adata_obj, organism="auto"):
    if organism == "auto":
        organism = infer_cell_cycle_organism(adata_obj)

    def filter_genes(adata_local, df, columns=None):
        if columns is None:
            columns = ["gene_name", "gene_id"]
        if isinstance(columns, str):
            columns = [columns]

        best_genes = []
        for col in columns:
            genes = [g for g in df[col] if g in adata_local.var_names]
            if len(genes) > len(best_genes):
                best_genes = genes

        if len(best_genes) == 0:
            raise ValueError("cell cycle genes not found in adata.var_names")
        return best_genes

    gene_map = get_cell_cycle_genes(organism)
    s_genes = filter_genes(adata_obj, gene_map.query("phase == 'S'"))
    g2m_genes = filter_genes(adata_obj, gene_map.query("phase == 'G2/M'"))
    sc.tl.score_genes_cell_cycle(adata_obj, s_genes=s_genes, g2m_genes=g2m_genes)


def _get_pcr_before_after(
    adata_pre,
    adata_post,
    batch_key,
    batch,
    embed,
    organism,
    n_comps=50,
):
    raw_sub = adata_pre[adata_pre.obs[batch_key] == batch].copy()
    post_sub = adata_post[adata_post.obs[batch_key] == batch].copy()

    _restore_scquartz_log_expression(raw_sub)

    if raw_sub.shape[0] != post_sub.shape[0]:
        raise ValueError(
            f"batch '{batch}' has unequal cells before/after integration: "
            f"{raw_sub.shape[0]} vs {post_sub.shape[0]}"
        )

    if ("S_score" not in raw_sub.obs) or ("G2M_score" not in raw_sub.obs):
        score_cell_cycle(raw_sub, organism=organism)

    covariate = raw_sub.obs[["S_score", "G2M_score"]]

    before = pc_regression(
        raw_sub.X,
        covariate,
        pca_var=None,
        n_comps=n_comps,
    )

    matrix_after = post_sub.obsm[embed] if embed is not None else post_sub.X
    after = pc_regression(
        matrix_after,
        covariate,
        pca_var=None,
        n_comps=n_comps,
    )

    return before, after


def cell_cycle_conservation(
    adata_pre,
    adata_post,
    batch_key,
    embed=None,
    agg_func=np.nanmean,
    organism="auto",
    n_comps=50,
):
    _check_adata(adata_pre)
    _check_adata(adata_post)

    if organism == "auto":
        organism = infer_cell_cycle_organism(adata_pre)

    if embed == "X_pca":
        embed = None

    scores = []
    for batch in adata_pre.obs[batch_key].unique():
        before, after = _get_pcr_before_after(
            adata_pre,
            adata_post,
            batch_key=batch_key,
            batch=batch,
            embed=embed,
            organism=organism,
            n_comps=n_comps,
        )

        if before <= 0:
            score = 0.0
        else:
            score = 1.0 - abs(after - before) / before
            if score < 0:
                score = 0.0

        scores.append(score)

    if agg_func is None:
        return scores
    return float(agg_func(scores)) if len(scores) > 0 else np.nan


# ============================================================
# HVG overlap
# ============================================================

def split_batches(adata_obj, batch_key, hvg=None):
    _ensure_category(adata_obj, batch_key)
    batch_categories = adata_obj.obs[batch_key].cat.categories

    adata_local = adata_obj
    if hvg is not None:
        adata_local = adata_local[:, hvg]

    out = []
    for cat in batch_categories:
        out.append(adata_local[adata_local.obs[batch_key] == cat].copy())
    return out


def precompute_hvg_batch(adata_obj, batch_key, features, n_hvg=500):
    adata_list = split_batches(adata_obj, batch_key, hvg=features)
    hvg_map = {}

    for adata_batch in adata_list:
        sc.pp.filter_genes(adata_batch, min_cells=1)
        n_hvg_tmp = np.minimum(n_hvg, int(0.5 * adata_batch.n_vars))
        hvg_df = sc.pp.highly_variable_genes(
            adata_batch,
            flavor="cell_ranger",
            n_top_genes=n_hvg_tmp,
            inplace=False,
        )
        batch_name = adata_batch.obs[batch_key].iloc[0]
        hvg_map[batch_name] = adata_batch.var.index[hvg_df["highly_variable"]]

    return hvg_map


def hvg_overlap(adata_pre, adata_post, batch_key, n_hvg=500, verbose=False):
    _ensure_category(adata_pre, batch_key)
    _ensure_category(adata_post, batch_key)

    hvg_post = adata_post.var_names
    adata_post_list = split_batches(adata_post, batch_key)
    hvg_pre_list = precompute_hvg_batch(adata_pre, batch_key, hvg_post, n_hvg=n_hvg)

    overlap = []
    for ad_post in adata_post_list:
        sc.pp.filter_genes(ad_post, min_cells=1)
        batch_name = ad_post.obs[batch_key].iloc[0]

        tmp_pre = hvg_pre_list[batch_name]
        n_hvg_tmp = len(tmp_pre)
        if verbose:
            print(f"batch {batch_name}, n_hvg={n_hvg_tmp}")

        hvg_df_post = sc.pp.highly_variable_genes(
            ad_post,
            flavor="cell_ranger",
            n_top_genes=n_hvg_tmp,
            inplace=False,
        )
        tmp_post = ad_post.var.index[hvg_df_post["highly_variable"]]

        n_hvg_real = np.minimum(len(tmp_pre), len(tmp_post))
        if n_hvg_real == 0:
            overlap.append(np.nan)
        else:
            overlap.append(len(set(tmp_pre).intersection(set(tmp_post))) / n_hvg_real)

    return float(np.nanmean(overlap)) if len(overlap) > 0 else np.nan


def hvg_batch(
    adata,
    batch_key,
    target_genes=2000,
    flavor="cell_ranger",
    n_bins=20,
    adataOut=False,
):
    _check_adata(adata)
    _check_batch(batch_key, adata.obs)
    _ensure_category(adata, batch_key)

    adata_hvg = adata if adataOut else adata.copy()
    n_batches = len(adata_hvg.obs[batch_key].cat.categories)

    sc.pp.highly_variable_genes(
        adata_hvg,
        flavor=flavor,
        n_top_genes=target_genes,
        n_bins=n_bins,
        batch_key=batch_key,
    )

    nbatch1 = adata_hvg.var["dispersions_norm"][
        adata_hvg.var.highly_variable_nbatches > (n_batches - 1)
    ]
    nbatch1 = nbatch1.sort_values(ascending=False)

    if len(nbatch1) > target_genes:
        hvg = list(nbatch1.index[:target_genes])
    else:
        hvg = list(nbatch1.index)
        not_n_batches = 1

        if not target_genes <= adata_hvg.n_vars:
            raise ValueError(
                f"Number of HVGs ({target_genes}) must be <= number of genes ({adata_hvg.n_vars})"
            )

        while len(hvg) < target_genes:
            target_diff = target_genes - len(hvg)
            tmp = adata_hvg.var["dispersions_norm"][
                adata_hvg.var.highly_variable_nbatches == (n_batches - not_n_batches)
            ]

            if len(tmp) == 0:
                not_n_batches += 1
                if not_n_batches > n_batches:
                    break
                continue

            if len(tmp) <= target_diff:
                hvg.extend(list(tmp.index))
                not_n_batches += 1
            else:
                tmp = tmp.sort_values(ascending=False)
                hvg.extend(list(tmp.index[:target_diff]))
                break

    if not adataOut:
        return hvg
    return adata_hvg[:, hvg].copy()


def prepare_evaluation_reference(
    adata_obj,
    already_preprocessed=False,
    n_pcs=50,
):
    """Build the common unintegrated reference used by evaluation metrics."""
    _check_adata(adata_obj)

    reference = ad.AnnData(
        X=adata_obj.X.copy(),
        obs=adata_obj.obs.copy(),
        var=pd.DataFrame(index=adata_obj.var_names.copy()),
    )
    reference.obs_names_make_unique()

    if not already_preprocessed:
        sc.pp.normalize_total(reference, target_sum=1e4)
        sc.pp.log1p(reference)

    max_pcs = min(reference.n_obs, reference.n_vars) - 1
    if max_pcs > 0:
        sc.tl.pca(
            reference,
            n_comps=min(n_pcs, max_pcs),
            use_highly_variable=False,
            svd_solver="arpack",
            random_state=0,
        )

    return reference


def prepare_moran_reference(
    adata_obj,
    batch_key,
    hvg_path=None,
    n_hvg=100,
):
    """Select Moran genes from the common preprocessed evaluation reference."""
    _check_adata(adata_obj)
    _check_batch(batch_key, adata_obj.obs)

    reference = ad.AnnData(
        X=adata_obj.X.copy(),
        obs=adata_obj.obs[[batch_key]].copy(),
        var=pd.DataFrame(index=adata_obj.var_names.copy()),
    )
    reference.obs_names_make_unique()

    hvg_file = Path(hvg_path) if hvg_path is not None else None
    if hvg_file is not None and hvg_file.exists():
        hvg_table = pd.read_csv(hvg_file)
        if "gene" not in hvg_table.columns:
            raise ValueError(f"Missing 'gene' column in {hvg_file}")
        hvgs = hvg_table["gene"].astype(str).tolist()
    else:
        hvgs = hvg_batch(
            reference,
            batch_key=batch_key,
            target_genes=n_hvg,
            flavor="cell_ranger",
        )
        if hvg_file is not None:
            temporary_path = Path(f"{hvg_file}.tmp")
            pd.DataFrame({"gene": hvgs}).to_csv(temporary_path, index=False)
            temporary_path.replace(hvg_file)

    if len(hvgs) != n_hvg or len(set(hvgs)) != len(hvgs):
        raise ValueError(
            f"Moran reference must contain exactly {n_hvg} unique genes"
        )
    missing = [gene for gene in hvgs if gene not in reference.var_names]
    if missing:
        raise ValueError(
            f"Moran reference genes are missing from the dataset: {missing[:5]}"
        )

    return reference[:, hvgs].copy()


# ============================================================
# Moran's I
# ============================================================

def _check_sanity(adata_obj, batch, hvg):
    _check_adata(adata_obj)
    _check_batch(batch, adata_obj.obs)
    if hvg is not None:
        if type(hvg) is not list:
            raise TypeError("HVG list is not a list")
        if not all(g in adata_obj.var.index for g in hvg):
            raise ValueError("Not all HVGs are in the adata object")


def _morans_i_fallback(adata_obj, hvgs):
    """Scanpy-compatible Moran's I for environments without ``sc.metrics``."""
    if hasattr(adata_obj, "obsp") and "connectivities" in adata_obj.obsp:
        graph = adata_obj.obsp["connectivities"]
    elif "neighbors" in adata_obj.uns and "connectivities" in adata_obj.uns["neighbors"]:
        graph = adata_obj.uns["neighbors"]["connectivities"]
    else:
        raise ValueError("Must run neighbors before Moran's I")
    graph = graph.astype(np.float64, copy=False)

    values = adata_obj[:, hvgs].X
    if issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)

    weight_sum = float(graph.sum())
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered * (graph @ centered), axis=0)
    return adata_obj.n_obs / weight_sum * numerator / denominator


def morans_i_score(adata_obj, hvgs):
    hvg_data = adata_obj[:, hvgs].X
    if issparse(hvg_data):
        mean = np.asarray(hvg_data.mean(axis=0)).ravel()
        mean_sq = np.asarray(hvg_data.multiply(hvg_data).mean(axis=0)).ravel()
        std_hvgs = np.sqrt(mean_sq - np.square(mean))
    else:
        std_hvgs = np.asarray(hvg_data.std(axis=0)).squeeze()

    hvgs_used = np.array(hvgs)[std_hvgs > 1e-8]
    if len(hvgs_used) == 0:
        adata_obj.var["morans_i"] = np.nan
        return

    if hasattr(sc, "metrics") and hasattr(sc.metrics, "morans_i"):
        morans = sc.metrics.morans_i(adata_obj[:, hvgs_used])
    else:
        morans = _morans_i_fallback(adata_obj, hvgs_used)
    morans = pd.Series(morans, index=hvgs_used)

    if not morans.between(-1, 1).all():
        raise ValueError("Computed Moran's I values are not in [-1, 1]")

    morans = np.maximum(morans, 0)
    adata_obj.var["morans_i"] = morans


def morans_i(
    adata_pre,
    adata_post,
    batch_key,
    n_hvg=100,
    embed="X_pca",
    compare_pre=False,
    hvgs=None,
    reference=None,
):
    if reference is None:
        _check_sanity(adata_pre, batch_key, hvgs)
        _check_sanity(adata_post, batch_key, hvgs)

        adata_post_local = adata_post.copy()
        _restore_scquartz_log_expression(adata_post_local)

        adata_pre_local = adata_pre.copy()
        _restore_scquartz_log_expression(adata_pre_local)
        _ensure_category(adata_pre_local, batch_key)

        if hvgs is None:
            hvgs = hvg_batch(
                adata_pre_local,
                batch_key=batch_key,
                target_genes=n_hvg,
                flavor="cell_ranger",
            )
    else:
        reference_hvgs = list(reference.var_names) if hvgs is None else hvgs
        _check_sanity(reference, batch_key, reference_hvgs)
        _check_adata(adata_post)
        if embed not in adata_post.obsm:
            raise KeyError(f"{embed} not in obsm")

        common = reference.obs_names[reference.obs_names.isin(adata_post.obs_names)]
        if len(common) == 0:
            raise ValueError("No common cells between Moran reference and post AnnData")

        post_index = adata_post.obs_names.get_indexer(common)
        if np.any(post_index < 0):
            raise ValueError("Could not align Moran reference cells to post AnnData")

        embedding = adata_post.obsm[embed]
        aligned_embedding = (
            embedding.iloc[post_index].to_numpy()
            if isinstance(embedding, pd.DataFrame)
            else embedding[post_index].copy()
        )
        adata_post_local = reference[common].copy()
        adata_post_local.obsm[embed] = aligned_embedding
        adata_pre_local = adata_post_local.copy()
        hvgs = reference_hvgs

    sc.pp.neighbors(adata_post_local, use_rep=embed)

    morans_i_score(adata_post_local, hvgs)
    score = adata_post_local.var["morans_i"].mean()

    if not compare_pre:
        return float(score)

    # Optional compare_pre branch mirrors scIB semantics when enabled.
    adatas_batches = {}
    for batch in adata_pre_local.obs[batch_key].unique():
        adata_batch = adata_pre_local[adata_pre_local.obs[batch_key] == batch, :].copy()
        sc.pp.highly_variable_genes(
            adata_batch,
            flavor="cell_ranger",
            batch_key=batch_key,
            n_top_genes=n_hvg,
        )

        adata_scale = adata_batch.copy()
        sc.pp.scale(adata_scale, max_value=10)
        sc.pp.pca(
            adata_scale,
            n_comps=15,
            use_highly_variable=True,
            svd_solver="arpack",
        )
        sc.pp.neighbors(adata_scale, n_pcs=15)

        adata_batch.uns["neighbors"] = adata_scale.uns["neighbors"]
        adata_batch.uns["pca"] = adata_scale.uns["pca"]
        adata_batch.obsm["X_pca"] = adata_scale.obsm["X_pca"]
        adata_batch.obsp["connectivities"] = adata_scale.obsp["connectivities"]
        adata_batch.obsp["distances"] = adata_scale.obsp["distances"]
        adatas_batches[batch] = adata_batch

    batch_scores = []
    for batch, adata_batch in adatas_batches.items():
        morans_i_score(adata_batch, hvgs)
        series = adata_batch.var["morans_i"]
        series.name = batch
        batch_scores.append(series)

    batch_scores = pd.concat(batch_scores, axis=1)
    morans_i_difference = batch_scores.max(axis=1) - adata_post_local.var["morans_i"]
    morans_i_difference = np.maximum(morans_i_difference, 0)
    return float(1 - morans_i_difference.mean())


# ============================================================
# Diffusion and kBET
# ============================================================

def diffusion_conn(adata, min_k=50, copy=True, max_iterations=26):
    if "neighbors" not in adata.uns:
        raise ValueError("`neighbors` not in adata. Please compute neighborhood graph")
    if "connectivities" not in adata.obsp:
        raise ValueError("`connectivities` not in adata.obsp")

    T = adata.obsp["connectivities"]
    T = sp_sparse.diags(1 / np.array([T.sum(1).max()] * T.shape[0])) * T
    M = T

    n_comp, labs = sp_sparse.csgraph.connected_components(
        adata.obsp["connectivities"],
        connection="strong",
    )

    if n_comp > 1:
        tab = pd.Series(labs).value_counts()
        small_comps = tab.index[tab < min_k]
        large_comp_mask = np.array(~pd.Series(labs).isin(small_comps))
    else:
        large_comp_mask = np.array([True] * M.shape[0])

    T_agg = T
    i = 2
    while ((M[large_comp_mask, :][:, large_comp_mask] > 0).sum(1).min() < min_k) and (i < max_iterations):
        T_agg *= T
        M += T_agg
        i += 1

    if (M[large_comp_mask, :][:, large_comp_mask] > 0).sum(1).min() < min_k:
        raise ValueError(
            "Could not create diffusion connectivities matrix with enough non-zero entries"
        )

    M.setdiag(0)

    if copy:
        adata_tmp = adata.copy()
        adata_tmp.uns["neighbors"].update({"diffusion_connectivities": M})
        return adata_tmp
    return M


def diffusion_nn(adata, k, max_iterations=26):
    if "neighbors" not in adata.uns:
        raise ValueError("`neighbors` not in adata. Please compute neighborhood graph")
    if "connectivities" not in adata.obsp:
        raise ValueError("`connectivities` not in adata.obsp")

    T = adata.obsp["connectivities"]
    T = sp_sparse.diags(1 / T.sum(1).A.ravel()) * T

    M = (T @ T).toarray()
    T_agg = T @ M
    M += T.toarray()
    M += T_agg

    i = 4
    while (np.count_nonzero(M, axis=1).min() < (k + 1)) and (i < max_iterations):
        T_agg = T @ T_agg
        M += T_agg
        i += 1

    if (M > 0).sum(1).min() < (k + 1):
        raise NeighborsError(f"could not find {k} nearest neighbors")

    np.fill_diagonal(M, 0)
    k_indices = np.zeros((M.shape[0], k), dtype=int)
    for idx, row in enumerate(M):
        k_indices[idx] = np.argpartition(row, -k)[-k:]

    return k_indices


def _kbet_single_py(
    batch,
    knn,
    k0=None,
    n_repeat=100,
    alpha=0.05,
    rng=None,
):
    """Python port of the observed rejection-rate test in R kBET."""
    batch_series = pd.Series(batch).reset_index(drop=True)
    n_obs = len(batch_series)
    if knn is None or n_obs <= 10:
        return np.nan

    batch_ref = batch_series.value_counts(normalize=True, sort=False)
    batch_levels = batch_ref.index.tolist()
    n_batches = len(batch_levels)
    if n_batches <= 1:
        return np.nan

    if k0 is None or k0 >= n_obs:
        k0 = int(np.floor(np.mean(batch_ref.to_numpy()) * n_obs / 4.0))
    k0 = int(k0)
    if k0 < 1:
        return np.nan

    # R kBET uses the first k0 - 1 neighbours plus the query itself.
    # Expected counts still use all k0 slots, including NA neighbour slots.
    reject_flags = np.full(n_obs, np.nan, dtype=float)
    expected = batch_ref.to_numpy(dtype=float) * k0
    for i in range(n_obs):
        idx = np.asarray(knn[i, : max(k0 - 1, 0)], dtype=float)
        idx = idx[np.isfinite(idx)].astype(int)
        idx = idx[(idx >= 0) & (idx < n_obs)]
        idx = np.concatenate((idx, np.asarray([i], dtype=int)))

        local = batch_series.iloc[idx]
        obs = (
            local.value_counts(normalize=False)
            .reindex(batch_levels, fill_value=0)
            .to_numpy(dtype=float)
        )
        stat = np.sum((obs - expected) ** 2 / expected)
        p_val = chi2.sf(stat, n_batches - 1)
        reject_flags[i] = float(p_val < alpha)

    test_size = int(np.ceil(0.1 * n_obs))
    if test_size < 25 and n_obs > 25:
        test_size = 25
    test_size = min(test_size, n_obs)
    if rng is None:
        rng = np.random.default_rng(0)

    observed = []
    for _ in range(n_repeat):
        sampled = rng.choice(n_obs, size=test_size, replace=False)
        if np.any(np.isfinite(reject_flags[sampled])):
            observed.append(float(np.nanmean(reject_flags[sampled])))

    if not observed:
        return np.nan
    return float(np.mean(observed))


def kBET(
    adata,
    batch_key,
    label_key,
    type_,
    embed=None,
    scaled=True,
    return_df=False,
    verbose=False,
):
    _check_adata(adata)
    _check_batch(batch_key, adata.obs)
    _check_batch(label_key, adata.obs)

    if type_ != "knn" and embed is not None:
        adata_tmp = sc.pp.neighbors(adata, n_neighbors=50, use_rep=embed, copy=True)
    else:
        adata_tmp = adata.copy()
        if "neighbors" not in adata_tmp.uns:
            sc.pp.neighbors(adata_tmp, n_neighbors=15, use_rep=embed)
        if "diffusion_connectivities" not in adata_tmp.uns["neighbors"]:
            adata_tmp = diffusion_conn(adata_tmp, min_k=50, copy=True)
        adata_tmp.obsp["connectivities"] = adata_tmp.uns["neighbors"][
            "diffusion_connectivities"
        ]

    size_max = 2**31 - 1

    counts = adata_tmp.obs.groupby(label_key).agg({label_key: "count", batch_key: "nunique"})
    labels = counts.query(f"{label_key}>=10 and {batch_key} > 1").index
    skipped = counts.index.difference(labels)

    kbet_scores = {"cluster": list(skipped), "kBET": [np.nan] * len(skipped)}
    rng = np.random.default_rng(0)

    for clus in labels:
        adata_sub = adata_tmp[adata_tmp.obs[label_key] == clus, :].copy()

        quarter_mean = int(np.floor(np.mean(adata_sub.obs[batch_key].value_counts()) / 4))
        k0 = int(np.min([70, np.max([10, quarter_mean])]))

        if k0 * adata_sub.n_obs >= size_max:
            k0 = int(np.floor(size_max / adata_sub.n_obs))

        n_comp, labs = sp_sparse.csgraph.connected_components(
            adata_sub.obsp["connectivities"],
            connection="strong",
        )

        if n_comp == 1:
            try:
                nn_index_tmp = diffusion_nn(adata_sub, k=k0).astype(float)
                score = _kbet_single_py(
                    batch=adata_sub.obs[batch_key],
                    knn=nn_index_tmp,
                    k0=k0,
                    rng=rng,
                )
            except NeighborsError:
                score = 1.0
        else:
            comp_size = pd.Series(labs).value_counts()
            comp_size_thresh = 3 * k0
            idx_nonan = np.flatnonzero(
                np.isin(labs, comp_size[comp_size >= comp_size_thresh].index)
            )

            if len(idx_nonan) / len(labs) >= 0.75:
                adata_sub_sub = adata_sub[idx_nonan, :].copy()
                nn_index_tmp = np.empty((adata_sub.n_obs, k0))
                nn_index_tmp[:] = np.nan
                try:
                    nn_index_tmp[idx_nonan] = diffusion_nn(adata_sub_sub, k=k0).astype(float)
                    score = _kbet_single_py(
                        batch=adata_sub.obs[batch_key],
                        knn=nn_index_tmp,
                        k0=k0,
                        rng=rng,
                    )
                except NeighborsError:
                    score = 1.0
            else:
                score = 1.0

        kbet_scores["cluster"].append(clus)
        kbet_scores["kBET"].append(score)

    kbet_scores = pd.DataFrame.from_dict(kbet_scores).reset_index(drop=True)

    if return_df:
        return kbet_scores

    final_score = np.nanmean(kbet_scores["kBET"])
    return float(1.0 - final_score) if scaled else float(final_score)


# ============================================================
# Graph-LISI (local implementation)
# ============================================================

def recompute_knn(adata, type_, use_rep="X_emb", graph_cache=None):
    cache_key = ("knn", id(adata), type_, use_rep)
    if graph_cache is not None and cache_key in graph_cache:
        return graph_cache[cache_key]

    if type_ == "embed":
        result = sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep, copy=True)
    elif type_ == "full":
        adata_tmp = adata.copy()
        if "X_pca" not in adata_tmp.obsm.keys():
            sc.pp.pca(adata_tmp, svd_solver="arpack")
        result = sc.pp.neighbors(adata_tmp, n_neighbors=15, copy=True)
    else:
        result = adata.copy()

    if graph_cache is not None:
        graph_cache[cache_key] = result
    return result


def _hbeta(dist_row, beta):
    dist_row = dist_row.copy()
    dist_row[np.isnan(dist_row)] = 0
    P = np.exp(-dist_row * beta)
    sumP = np.sum(P)
    if sumP == 0:
        H = 0
        P = np.zeros(len(dist_row))
    else:
        H = np.log(sumP) + beta * np.sum(dist_row * P) / sumP
        P /= sumP
    return H, P


def _convert_to_one_hot(vector, num_classes=None):
    if num_classes is None:
        num_classes = np.max(vector) + 1
    return np.eye(num_classes)[vector]


def _topk_shortest_path_neighbors(connectivities, start, k):
    indptr = connectivities.indptr
    indices = connectivities.indices
    data = connectivities.data

    best_dist = {int(start): 0.0}
    visited = set()
    heap = [(0.0, int(start))]

    out_idx = []
    out_dist = []

    while heap and len(out_idx) < k:
        dist_u, u = heapq.heappop(heap)

        if u in visited:
            continue
        if dist_u > best_dist.get(u, np.inf):
            continue

        visited.add(u)

        if u != start:
            out_idx.append(u)
            out_dist.append(dist_u)
            if len(out_idx) >= k:
                break

        row_start = indptr[u]
        row_end = indptr[u + 1]

        for ptr in range(row_start, row_end):
            v = int(indices[ptr])
            w = float(data[ptr])
            new_d = dist_u + w
            if new_d < best_dist.get(v, np.inf):
                best_dist[v] = new_d
                heapq.heappush(heap, (new_d, v))

    return np.asarray(out_idx, dtype=int), np.asarray(out_dist, dtype=float)


def _compute_simpson_index_graph(
    connectivities,
    batch_labels,
    n_batches,
    n_neighbors=90,
    perplexity=30,
    subset_idx=None,
    tol=1e-5,
    graph_neighbors=None,
):
    if subset_idx is None:
        subset_idx = np.arange(connectivities.shape[0])

    if graph_neighbors is None:
        graph_neighbors = [
            _topk_shortest_path_neighbors(connectivities, int(cell_id), n_neighbors)
            for cell_id in subset_idx
        ]

    simpson = np.zeros(len(subset_idx), dtype=float)
    logU = np.log(perplexity)

    for i, (knn_idx, distances) in enumerate(graph_neighbors):

        if len(knn_idx) < n_neighbors:
            simpson[i] = 1.0
            continue

        beta = 1.0
        betamin = -np.inf
        betamax = np.inf

        H, P = _hbeta(distances, beta)
        Hdiff = H - logU
        tries = 0

        while np.logical_and(np.abs(Hdiff) > tol, tries < 50):
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            H, P = _hbeta(distances, beta)
            Hdiff = H - logU
            tries += 1

        if H == 0:
            simpson[i] = -1
            continue

        batch = batch_labels[knn_idx]
        one_hot = _convert_to_one_hot(batch, n_batches)
        sumP = np.matmul(P, one_hot)
        simpson[i] = np.dot(sumP, sumP)

    return simpson


def lisi_graph_py(
    adata,
    obs_key,
    n_neighbors=90,
    perplexity=None,
    subsample=None,
    verbose=False,
    graph_cache=None,
):
    if "neighbors" not in adata.uns:
        raise AttributeError("Key 'neighbors' not found. Please compute a kNN graph first")

    labels = adata.obs[obs_key]
    if not pd.api.types.is_categorical_dtype(labels):
        labels = labels.astype("category")

    batch_labels = labels.cat.codes.values
    n_batches = labels.nunique()

    if perplexity is None or perplexity >= n_neighbors:
        perplexity = int(np.floor(n_neighbors / 3))
    perplexity = max(perplexity, 1)

    subset_pct = 100
    if subsample is not None:
        subset_pct = int(subsample)

    n_obs = adata.n_obs
    if subset_pct >= 100:
        subset_idx = np.arange(n_obs)
    else:
        n_subset = max(1, int(np.floor(n_obs * (subset_pct / 100.0))))
        rng = np.random.default_rng(0)
        subset_idx = np.sort(rng.choice(n_obs, size=n_subset, replace=False))

    cache_key = ("lisi_paths", id(adata), n_neighbors, subset_pct)
    cached = None if graph_cache is None else graph_cache.get(cache_key)
    if cached is None:
        connectivities = adata.obsp["connectivities"].tocsr().copy()
        large_enough = connectivities.data >= 3e-308
        if verbose:
            n_too_small = np.sum(~large_enough)
            if n_too_small > 0:
                print(
                    f"{n_too_small} connectivities are smaller than 3e-308 "
                    "and will be clipped"
                )
        connectivities.data[~large_enough] = 3e-308
        graph_neighbors = [
            _topk_shortest_path_neighbors(
                connectivities,
                int(cell_id),
                n_neighbors,
            )
            for cell_id in subset_idx
        ]
        cached = (connectivities, subset_idx, graph_neighbors)
        if graph_cache is not None:
            graph_cache[cache_key] = cached
    else:
        connectivities, subset_idx, graph_neighbors = cached

    simpson = _compute_simpson_index_graph(
        connectivities=connectivities,
        batch_labels=batch_labels,
        n_batches=n_batches,
        n_neighbors=n_neighbors,
        perplexity=perplexity,
        subset_idx=subset_idx,
        graph_neighbors=graph_neighbors,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = 1.0 / simpson
    scores[~np.isfinite(scores)] = np.nan
    return scores


def ilisi_graph(
    adata,
    batch_key,
    type_,
    use_rep="X_emb",
    k0=90,
    subsample=None,
    scale=True,
    verbose=False,
    graph_cache=None,
):
    _check_adata(adata)
    _check_batch(batch_key, adata.obs)

    nbatches = adata.obs[batch_key].nunique()
    if scale and nbatches <= 1:
        return np.nan

    adata_tmp = recompute_knn(
        adata,
        type_,
        use_rep,
        graph_cache=graph_cache,
    )
    ilisi_score = lisi_graph_py(
        adata=adata_tmp,
        obs_key=batch_key,
        n_neighbors=k0,
        perplexity=None,
        subsample=subsample,
        verbose=verbose,
        graph_cache=graph_cache,
    )

    ilisi = np.nanmedian(ilisi_score)

    if scale:
        ilisi = (ilisi - 1.0) / (nbatches - 1.0)

    return float(ilisi)


def cilisi_graph(
    adata,
    batch_key,
    label_key,
    use_rep="X_emb",
    k0=90,
    subsample=None,
    scale=True,
    verbose=False,
):
    """Cell-type-adjusted iLISI from the Metrics Matter implementation."""
    _check_adata(adata)
    _check_batch(batch_key, adata.obs)
    _check_batch(label_key, adata.obs)

    weighted_scores = []
    total_cells = 0
    for label in adata.obs[label_key].unique():
        adata_group = adata[adata.obs[label_key] == label].copy()
        adata_group.obs[batch_key] = (
            adata_group.obs[batch_key]
            .astype("category")
            .cat.remove_unused_categories()
        )
        score = ilisi_graph(
            adata_group,
            batch_key=batch_key,
            type_="knn",
            use_rep=use_rep,
            k0=90,
            subsample=None,
            scale=scale,
            verbose=verbose,
        )
        weighted_scores.append(score * adata_group.n_obs)
        total_cells += adata_group.n_obs

    if total_cells == 0:
        return np.nan
    return float(np.nansum(weighted_scores) / total_cells)


def clisi_graph(
    adata,
    label_key,
    type_,
    use_rep="X_emb",
    batch_key=None,
    k0=90,
    subsample=None,
    scale=True,
    verbose=False,
    graph_cache=None,
):
    if batch_key is not None:
        warnings.warn("'batch_key' is deprecated for cLISI and ignored", stacklevel=1)

    _check_adata(adata)
    _check_batch(label_key, adata.obs)

    adata_tmp = recompute_knn(
        adata,
        type_,
        use_rep,
        graph_cache=graph_cache,
    )
    scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key=label_key,
        n_neighbors=k0,
        perplexity=None,
        subsample=subsample,
        verbose=verbose,
        graph_cache=graph_cache,
    )

    clisi = np.nanmedian(scores)

    if scale:
        nlabs = adata.obs[label_key].nunique()
        if nlabs <= 1:
            return 1.0
        clisi = (nlabs - clisi) / (nlabs - 1.0)

    return float(clisi)


# ============================================================
# Trajectory conservation
# ============================================================

def get_root(adata_pre, adata_post, ct_key, pseudotime_key="dpt_pseudotime", dpt_dim=3):
    _, adata_post.obs["neighborhood"] = connected_components(
        csgraph=adata_post.obsp["connectivities"],
        directed=False,
        return_labels=True,
    )

    start_clust = adata_pre.obs.groupby(ct_key)[pseudotime_key].mean().idxmin()
    min_dpt = adata_pre.obs[adata_pre.obs[ct_key] == start_clust].index

    which_max_neigh = (
        adata_post.obs["neighborhood"]
        == adata_post.obs["neighborhood"].value_counts().idxmax()
    )
    min_dpt = [x for x in min_dpt if x in adata_post.obs[which_max_neigh].index]

    adata_post_ti = adata_post[which_max_neigh]
    min_dpt = [adata_post_ti.obs_names.get_loc(i) for i in min_dpt]

    sc.tl.diffmap(adata_post_ti)

    min_dpt_cell = np.zeros(len(min_dpt))
    for dim in np.arange(dpt_dim):
        diffmap_mean = adata_post_ti.obsm["X_diffmap"][:, dim].mean()
        diffmap_min_dpt = adata_post_ti.obsm["X_diffmap"][min_dpt, dim]

        if len(diffmap_min_dpt) == 0:
            raise RootCellError("No root cell in largest component")

        if len(diffmap_min_dpt) > 0 and diffmap_min_dpt.mean() < diffmap_mean:
            opt = np.argmin
        else:
            opt = np.argmax

        min_dpt_cell[opt(diffmap_min_dpt)] += 1

    return min_dpt[np.argmax(min_dpt_cell)], adata_post_ti


def trajectory_conservation(
    adata_pre,
    adata_post,
    label_key,
    pseudotime_key="dpt_pseudotime",
    batch_key=None,
):
    if pseudotime_key not in adata_pre.obs:
        raise KeyError(
            f"{pseudotime_key} missing in pre-integration adata.obs; trajectory metric needs precomputed pseudotime"
        )

    cell_subset = adata_pre.obs.index[adata_pre.obs[pseudotime_key].notnull()]
    adata_pre_ti = adata_pre[cell_subset]
    adata_post_ti = adata_post[cell_subset]

    try:
        iroot, adata_post_ti2 = get_root(
            adata_pre_ti,
            adata_post_ti,
            label_key,
            pseudotime_key,
        )
    except RootCellError:
        return 0.0

    adata_post_ti2.uns["iroot"] = iroot
    sc.tl.dpt(adata_post_ti2)

    adata_post_ti2.obs.loc[
        adata_post_ti2.obs["dpt_pseudotime"] > 1,
        "dpt_pseudotime",
    ] = 0

    adata_post_ti.obs["dpt_pseudotime"] = 0
    adata_post_ti.obs["dpt_pseudotime"] = adata_post_ti2.obs["dpt_pseudotime"]
    adata_post_ti.obs["dpt_pseudotime"].fillna(0, inplace=True)

    if batch_key is None:
        pseudotime_before = adata_pre_ti.obs[pseudotime_key]
        pseudotime_after = adata_post_ti.obs["dpt_pseudotime"]
        corr = pseudotime_before.corr(pseudotime_after, "spearman")
        return float((corr + 1.0) / 2.0)

    _check_batch(batch_key, adata_pre.obs)
    _check_batch(batch_key, adata_post.obs)

    if not np.array_equal(adata_post_ti.obs[batch_key], adata_pre_ti.obs[batch_key]):
        raise ValueError("Batch columns do not match between pre and post trajectory subsets")

    corr = []
    for batch in adata_pre_ti.obs[batch_key].unique():
        before = adata_pre_ti.obs[adata_pre_ti.obs[batch_key] == batch][pseudotime_key]
        after = adata_post_ti.obs[adata_post_ti.obs[batch_key] == batch]["dpt_pseudotime"]
        corr.append(before.corr(after, "spearman"))

    return float((np.nanmean(corr) + 1.0) / 2.0)


# ============================================================
# Lightweight BatchKL helper kept for compatibility
# ============================================================

def calculate_batch_kl_py(
    embedding,
    batch_labels,
    n_cells=100,
    n_neighbors=100,
    replicates=200,
):
    np.random.seed(42)
    batch_series = pd.Series(batch_labels)
    batch_counts = batch_series.value_counts(normalize=True).sort_index()
    p_population = batch_counts.values
    unique_batches = batch_counts.index.tolist()

    k = min(5 * len(unique_batches), n_neighbors)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1).fit(embedding)

    kl_replicates = []
    for _ in range(replicates):
        boot_idx = np.random.choice(embedding.shape[0], size=n_cells, replace=False)
        _, indices = nn.kneighbors(embedding[boot_idx])
        kl_samples = [
            entropy(
                batch_series.iloc[idx]
                .value_counts(normalize=True)
                .reindex(unique_batches, fill_value=0)
                .values
                + 1e-10,
                p_population + 1e-10,
                base=2,
            )
            for idx in indices
        ]
        kl_replicates.append(np.mean(kl_samples))
    return float(np.mean(kl_replicates))


# ============================================================
# Local scIB metrics wrapper (equivalent to metrics_all defaults)
# ============================================================

def metrics_all_local(
    adata,
    adata_int,
    batch_key,
    label_key,
    embed="X_pca",
    cluster_key="cluster",
    cluster_nmi=None,
    nmi_method="arithmetic",
    si_metric="euclidean",
    organism="auto",
    n_isolated=None,
    subsample=0.5,
    n_cores=1,
    type_=None,
    verbose=False,
    moran_reference=None,
):
    _check_adata(adata)
    _check_batch(batch_key, adata.obs)
    _check_batch(label_key, adata.obs)

    _check_adata(adata_int)
    _check_batch(batch_key, adata_int.obs)
    _check_batch(label_key, adata_int.obs)

    # Match scIB.metrics: optimize clustering whenever NMI/ARI are used.
    if cluster_key in adata_int.obs.columns:
        if verbose:
            print(f"Using existing clustering: {cluster_key}")
    else:
        _, _, nmi_all = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=nmi_score,
            use_rep=embed,
            force=True,
            verbose=verbose,
            return_all=True,
        )

    if cluster_nmi is not None:
        nmi_all.to_csv(cluster_nmi, header=False)

    lisi_subsample = None if subsample is None else (subsample * 100)
    silhouette_cache = {}
    lisi_cache = {}

    results = {
        "NMI_cluster/label": _safe_metric(
            "NMI_cluster/label",
            nmi_score,
            adata_int,
            cluster_key=cluster_key,
            label_key=label_key,
            implementation=nmi_method,
        ),
        "ARI_cluster/label": _safe_metric(
            "ARI_cluster/label",
            ari_score,
            adata_int,
            cluster_key=cluster_key,
            label_key=label_key,
        ),
        "ASW_label": _safe_metric(
            "ASW_label",
            silhouette_label,
            adata_int,
            label_key=label_key,
            embed=embed,
            metric=si_metric,
            scale=True,
            silhouette_cache=silhouette_cache,
        ),
        "ASW_label/batch": _safe_metric(
            "ASW_label/batch",
            silhouette_batch,
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            metric=si_metric,
            return_all=False,
            verbose=False,
        ),
        "BRAS": _safe_metric(
            "BRAS",
            bras,
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
        ),
        "PCR_batch": _safe_metric(
            "PCR_batch",
            pcr_comparison,
            adata,
            adata_int,
            covariate=batch_key,
            embed=embed,
            scale=True,
        ),
        "cell_cycle_conservation": _safe_metric(
            "cell_cycle_conservation",
            cell_cycle_conservation,
            adata,
            adata_int,
            batch_key=batch_key,
            embed=embed,
            agg_func=np.mean,
            organism=organism,
        ),
        "isolated_label_F1": _safe_metric(
            "isolated_label_F1",
            isolated_labels_metric,
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=True,
            cluster_key=cluster_key,
            iso_threshold=n_isolated,
            verbose=False,
        ),
        "isolated_label_silhouette": _safe_metric(
            "isolated_label_silhouette",
            isolated_labels_metric,
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=False,
            iso_threshold=n_isolated,
            scale=True,
            verbose=False,
            silhouette_cache=silhouette_cache,
        ),
        "graph_conn": _safe_metric(
            "graph_conn",
            graph_connectivity,
            adata_int,
            label_key=label_key,
        ),
        "kBET": _safe_metric(
            "kBET",
            kBET,
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=type_,
            embed=embed,
            scaled=True,
            verbose=verbose,
        ),
        "iLISI": _safe_metric(
            "iLISI",
            ilisi_graph,
            adata_int,
            batch_key=batch_key,
            type_=type_,
            use_rep=embed,
            subsample=lisi_subsample,
            scale=True,
            verbose=verbose,
            graph_cache=lisi_cache,
        ),
        "CiLISI": _safe_metric(
            "CiLISI",
            cilisi_graph,
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            use_rep=embed,
            subsample=None,
            scale=True,
            verbose=verbose,
        ),
        "cLISI": _safe_metric(
            "cLISI",
            clisi_graph,
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=type_,
            use_rep=embed,
            subsample=lisi_subsample,
            scale=True,
            verbose=verbose,
            graph_cache=lisi_cache,
        ),
        "hvg_overlap": _safe_metric(
            "hvg_overlap",
            hvg_overlap,
            adata,
            adata_int,
            batch_key=batch_key,
        ),
        "trajectory": _safe_metric(
            "trajectory",
            trajectory_conservation,
            adata,
            adata_int,
            label_key=label_key,
        ),
        "morans_i": _safe_metric(
            "morans_i",
            morans_i,
            adata,
            adata_int,
            batch_key=batch_key,
            embed=embed,
            reference=moran_reference,
        ),
    }

    return results


# ============================================================
# Main entry used by pipeline
# ============================================================

def evaluate_embedding_scib(
    ad_tmp,
    embed_key="X_supcon",
    batch_key="batch",
    celltype_key="celltype",
    leiden_resolution=0.1,
    random_state=42,
    precomputed_cluster_key=None,
    adata_pre=None,
    scib_type=None,
    scib_subsample=0.5,
    scib_n_cores=1,
    scib_organism="auto",
    scib_verbose=False,
    moran_reference=None,
):
    """
    Local implementation of scIB metrics_all (without scIB API calls).

    Notes:
    - Uses scIB-equivalent defaults and metric names.
    - Keeps legacy aliases for existing downstream scripts.
    - Custom metrics that are not in scIB are left untouched.
    """

    if embed_key not in ad_tmp.obsm:
        raise KeyError(f"Embedding '{embed_key}' not found in adata.obsm")

    adata_pre_local = adata_pre if adata_pre is not None else ad_tmp
    adata_pre_aligned, adata_post_aligned = _align_pre_post(
        adata_pre_local,
        ad_tmp,
    )

    results = metrics_all_local(
        adata=adata_pre_aligned,
        adata_int=adata_post_aligned,
        batch_key=batch_key,
        label_key=celltype_key,
        embed=embed_key,
        cluster_key=(
            precomputed_cluster_key
            if precomputed_cluster_key is not None
            else "cluster"
        ),
        nmi_method="arithmetic",
        si_metric="euclidean",
        organism=scib_organism,
        n_isolated=None,
        subsample=scib_subsample,
        n_cores=scib_n_cores,
        type_=scib_type,
        verbose=scib_verbose,
        moran_reference=moran_reference,
    )

    # Legacy aliases for existing plotting / summaries.
    results["NMI"] = results["NMI_cluster/label"]
    results["ARI"] = results["ARI_cluster/label"]
    results["ASW_bio"] = results["ASW_label"]
    results["ASW_batch"] = results["ASW_label/batch"]
    results["GraphConn_celltype"] = results["graph_conn"]

    results["scib_cluster_key"] = precomputed_cluster_key if precomputed_cluster_key is not None else "cluster"
    results["scib_type"] = scib_type
    results["scib_subsample"] = scib_subsample
    results["scib_n_cores"] = scib_n_cores
    results["scib_organism"] = scib_organism
    return results
