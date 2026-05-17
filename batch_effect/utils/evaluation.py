import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples
)
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy.sparse.csgraph import connected_components
from harmonypy import compute_lisi

def compute_overcorrection_from_adata(
    adata,
    embed_key="X_pca",
    celltype_col="celltype",
    n_neighbors=100,
    n_pools=100,
    n_samples_per_pool=100,
    seed=124,
    baseline=None
):
    """
    Compute Over-correction Score using pooled sampling method.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing obs and embeddings
    embed_key : str
        Key for embedding in adata.obsm
    celltype_col : str
        obs column containing cell types
    n_neighbors : int
        Number of neighbors to consider
    n_pools : int
        Number of pooling iterations
    n_samples_per_pool : int
        Number of samples per pool
    seed : int
        Random seed for reproducibility
    baseline : float or None
        If provided, returns (current_score - baseline)
    
    Returns
    -------
    float
        Over-correction score (or normalized score if baseline provided)
    """
    
    # 1. Extract Embedding
    if embed_key in adata.obsm.keys():
        Z = adata.obsm[embed_key]
    elif embed_key == "X":
        Z = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    else:
        raise ValueError(f"Embedding '{embed_key}' not found in adata.obsm")

    # 2. Prepare label data
    celltype_series = pd.Series(adata.obs[celltype_col]).reset_index(drop=True)
    celltype_values = celltype_series.values
    celltype_dict = celltype_series.value_counts().to_dict()
    
    n_cells = Z.shape[0]
    n_neighbors = min(n_neighbors, n_cells - 1)
    
    # 3. Compute neighbor matrix (exclude self)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=-1)
    nne.fit(Z)
    kmatrix = nne.kneighbors_graph(Z) - sp_sparse.identity(n_cells)
    
    # 4. Pooling random sampling
    np.random.seed(seed)
    total_score = 0
    
    for t in range(n_pools):
        indices = np.random.choice(np.arange(n_cells), size=min(n_samples_per_pool, n_cells), replace=False)
        pool_purity = []
        
        for i in indices:
            neighbor_indices = kmatrix[i].nonzero()[1]
            # Capping: limit neighbors based on cell type count
            max_compare = min(celltype_dict[celltype_values[i]], n_neighbors)
            
            # Calculate proportion of same-type neighbors
            if len(neighbor_indices) > 0 and max_compare > 0:
                is_same_type = (celltype_values[neighbor_indices[:max_compare]] == celltype_values[i])
                pool_purity.append(np.mean(is_same_type))
            else:
                pool_purity.append(1.0)  # No neighbors = assume pure
            
        total_score += np.mean(pool_purity)

    # 5. Calculate final raw OC (1 - average purity)
    raw_oc_mean = 1 - (total_score / float(n_pools))

    # 6. Return normalized if baseline provided
    if baseline is not None:
        oc_norm = float(raw_oc_mean - baseline)
        print(f"[OC] baseline={baseline:.4f}, integrated={raw_oc_mean:.4f}, normalized={oc_norm:.4f}")
        return oc_norm
    
    print(f"[OC] raw OC = {raw_oc_mean:.4f}")
    return float(raw_oc_mean)


def calculate_batch_kl_py(embedding, batch_labels, n_cells=100, n_neighbors=100, replicates=200):
    """計算 Batch KL Divergence (原始散度值，越低越好)"""
    np.random.seed(42)
    batch_series = pd.Series(batch_labels)
    batch_counts = batch_series.value_counts(normalize=True).sort_index()
    p_population = batch_counts.values
    unique_batches = batch_counts.index.tolist()
    
    k = min(5 * len(unique_batches), n_neighbors)
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1).fit(embedding)
    
    kl_replicates = []
    for _ in range(replicates):
        boot_idx = np.random.choice(embedding.shape[0], size=n_cells, replace=False)
        _, indices = nn.kneighbors(embedding[boot_idx])
        kl_samples = [entropy(batch_series.iloc[idx].value_counts(normalize=True).reindex(unique_batches, fill_value=0).values + 1e-10, 
                             p_population + 1e-10, base=2) for idx in indices]
        kl_replicates.append(np.mean(kl_samples))
    return np.mean(kl_replicates)


def calculate_lisi(embedding, adata_obs, batch_key, celltype_key):
    """計算原始 iLISI (有效批次數)"""
    lisi_res = compute_lisi(embedding, adata_obs, [batch_key, celltype_key])
    lisi_raw = lisi_res[:, 0]
    ilisi_std = np.median(lisi_raw)
    
    return ilisi_std


def evaluate_embedding_scib(
    ad_tmp,
    embed_key="X_supcon",
    batch_key="batch",
    celltype_key="celltype",
    leiden_resolution=0.1,
    random_state=42,
    compute_oc=False,
    precomputed_cluster_key=None,
):
    """
    原生 Python 版的 scIB 評估管線 (無須安裝 scib-metrics/JAX)
    底層數學邏輯對齊 scIB 官方定義
    """

    results = {}

    embedding = ad_tmp.obsm[embed_key]
    if hasattr(embedding, "toarray"):
        embedding = embedding.toarray()

    y = ad_tmp.obs[celltype_key].astype(str).values
    b = ad_tmp.obs[batch_key].astype(str).values

    # ============================================================
    # 統一 neighbors graph
    # 確保 clustering / graph connectivity 使用同一 embedding
    # ============================================================

    need_neighbors = (
        "neighbors" not in ad_tmp.uns
        or ad_tmp.uns["neighbors"]["params"].get("use_rep", None) != embed_key
    )

    if need_neighbors:
        sc.pp.neighbors(
            ad_tmp,
            n_neighbors=15,
            use_rep=embed_key,
        )

    # ============================================================
    # 聚類標籤 (NMI / ARI)
    # ============================================================

    if (
        precomputed_cluster_key is not None
        and precomputed_cluster_key in ad_tmp.obs
    ):
        cluster_labels = (
            ad_tmp.obs[precomputed_cluster_key]
            .astype(str)
            .values
        )

    else:
        sc.tl.leiden(
            ad_tmp,
            resolution=leiden_resolution,
            random_state=random_state,
            key_added="eval_leiden",
        )

        cluster_labels = (
            ad_tmp.obs["eval_leiden"]
            .astype(str)
            .values
        )

    # ============================================================
    # Biological conservation
    # ============================================================

    results["NMI"] = float(
        normalized_mutual_info_score(y, cluster_labels)
    )

    results["ARI"] = float(
        adjusted_rand_score(y, cluster_labels)
    )

    # ASW bio
    sil_bio = silhouette_samples(embedding, y)

    # scIB scaling
    sil_bio_scaled = (sil_bio + 1.0) / 2.0

    results["ASW_bio"] = float(
        pd.DataFrame({
            "ct": y,
            "sil": sil_bio_scaled
        })
        .groupby("ct")["sil"]
        .mean()
        .mean()
    )

    results["AVG_bio"] = float(
        np.mean([
            results["NMI"],
            results["ARI"],
            results["ASW_bio"],
        ])
    )

    # ============================================================
    # Batch correction
    # ============================================================

    # ------------------------------------------------------------
    # ASW batch
    # scIB:
    # 在每個 cell type 內計算 batch silhouette
    # 使用 1 - abs(score)
    # ------------------------------------------------------------

    batch_asw_list = []

    for cell_type in np.unique(y):

        idx = np.where(y == cell_type)[0]

        # 若只有單一 batch 則跳過
        if len(np.unique(b[idx])) < 2:
            continue

        sil_batch = silhouette_samples(
            embedding[idx],
            b[idx]
        )

        sil_batch_scaled = 1.0 - np.abs(sil_batch)

        batch_asw_list.append(
            np.mean(sil_batch_scaled)
        )

    results["ASW_batch"] = (
        float(np.mean(batch_asw_list))
        if batch_asw_list
        else 0.0
    )

    # ------------------------------------------------------------
    # Graph Connectivity
    # scIB:
    # 每個 cell type 子圖的最大連通分量比例
    # ------------------------------------------------------------

    connectivities = ad_tmp.obsp["connectivities"]

    graph_conn_list = []

    for cell_type in np.unique(y):

        idx = np.where(y == cell_type)[0]

        if len(idx) <= 1:
            continue

        # cell type subgraph
        subgraph = connectivities[idx, :][:, idx]

        # 避免 sparse edge cases
        subgraph = subgraph.tocsr()

        n_components, labels_comp = connected_components(
            subgraph,
            directed=False,
        )

        _, counts = np.unique(
            labels_comp,
            return_counts=True,
        )

        lcc_ratio = counts.max() / len(idx)

        graph_conn_list.append(lcc_ratio)

    results["GraphConn_celltype"] = (
        float(np.mean(graph_conn_list))
        if graph_conn_list
        else 0.0
    )

    # ------------------------------------------------------------
    # iLISI
    # 修正為 scIB normalized version
    # ------------------------------------------------------------

    ilisi_res = compute_lisi(
        embedding,
        ad_tmp.obs,
        [batch_key],
    )

    ilisi_raw = float(
        np.median(ilisi_res[:, 0])
    )

    n_batches = len(np.unique(b))

    if n_batches > 1:
        ilisi_scaled = (
            (ilisi_raw - 1.0)
            / (n_batches - 1.0)
        )
    else:
        ilisi_scaled = 1.0

    results["iLISI"] = float(ilisi_scaled)

    # ------------------------------------------------------------
    # AVG batch
    # ------------------------------------------------------------

    results["AVG_batch"] = float(
        np.mean([
            results["ASW_batch"],
            results["GraphConn_celltype"],
        ])
    )

    # ============================================================
    # Over-correction
    # ============================================================

    if compute_oc:

        if "X_pca" not in ad_tmp.obsm:
            sc.pp.pca(ad_tmp)

        oc_raw = compute_overcorrection_from_adata(
            ad_tmp,
            embed_key="X_pca",
            celltype_col=celltype_key,
        )

        results["OverCorrection"] = (
            compute_overcorrection_from_adata(
                ad_tmp,
                embed_key=embed_key,
                celltype_col=celltype_key,
                baseline=oc_raw,
            )
        )

    else:
        results["OverCorrection"] = np.nan

    return results