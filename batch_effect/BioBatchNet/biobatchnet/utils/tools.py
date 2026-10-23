import scanpy as sc
import scib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, balanced_accuracy_score, r2_score
import anndata as ad


def visualization(bio_z, batch_z, batch_labels, cell_types, save_path, batch_key='BATCH', label_key='celltype'):
    """Create 2x2 UMAP visualization for bio_z and batch_z with batch and celltype coloring."""
    # Create adata for bio_z
    adata_bio = ad.AnnData(bio_z)
    adata_bio.obs[batch_key] = pd.Categorical(batch_labels)
    adata_bio.obs[label_key] = pd.Categorical(cell_types)

    # Create adata for batch_z
    adata_batch = ad.AnnData(batch_z)
    adata_batch.obs[batch_key] = pd.Categorical(batch_labels)
    adata_batch.obs[label_key] = pd.Categorical(cell_types)

    # Subsample both
    adata_bio = sc.pp.subsample(adata_bio, fraction=0.3, random_state=42, copy=True)
    adata_batch = sc.pp.subsample(adata_batch, fraction=0.3, random_state=42, copy=True)

    # Compute UMAP for both
    sc.pp.neighbors(adata_bio)
    sc.tl.umap(adata_bio)
    sc.pp.neighbors(adata_batch)
    sc.tl.umap(adata_batch)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Get unique categories and create color maps
    batch_cats = sorted(adata_bio.obs[batch_key].unique())
    cell_cats = sorted(adata_bio.obs[label_key].unique())

    batch_palette = sc.pl.palettes.default_20 if len(batch_cats) <= 20 else sc.pl.palettes.default_102
    cell_palette = sc.pl.palettes.default_20 if len(cell_cats) <= 20 else sc.pl.palettes.default_102

    batch_colors = {cat: batch_palette[i] for i, cat in enumerate(batch_cats)}
    cell_colors = {cat: cell_palette[i] for i, cat in enumerate(cell_cats)}

    # Row 1: z_bio
    sc.pl.umap(adata_bio, color=batch_key, ax=axes[0, 0], show=False, legend_loc=None,
               frameon=False, title='z_bio - Batch', palette=[batch_colors[c] for c in batch_cats])
    sc.pl.umap(adata_bio, color=label_key, ax=axes[0, 1], show=False, legend_loc=None,
               frameon=False, title='z_bio - Cell Type', palette=[cell_colors[c] for c in cell_cats])

    # Row 2: z_batch
    sc.pl.umap(adata_batch, color=batch_key, ax=axes[1, 0], show=False, legend_loc=None,
               frameon=False, title='z_batch - Batch', palette=[batch_colors[c] for c in batch_cats])
    sc.pl.umap(adata_batch, color=label_key, ax=axes[1, 1], show=False, legend_loc=None,
               frameon=False, title='z_batch - Cell Type', palette=[cell_colors[c] for c in cell_cats])

    # Add legends
    batch_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                            markerfacecolor=batch_colors[cat], markersize=8)
                     for cat in batch_cats]
    cell_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                           markerfacecolor=cell_colors[cat], markersize=8)
                    for cat in cell_cats]

    # Add legends to bottom row
    axes[1, 0].legend(handles=batch_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                      ncol=min(len(batch_cats), 6), fontsize=8, frameon=False)
    axes[1, 1].legend(handles=cell_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                      ncol=min(len(cell_cats), 4), fontsize=8, frameon=False)

    plt.subplots_adjust(bottom=0.15, hspace=0.25, wspace=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualization_with_leiden(adata, save_path, batch_key='BATCH', label_key='celltype', seed=42):
    """Create 3-panel UMAP: celltype, batch, and leiden cluster for eval phase.

    Expects neighbors and leiden to already be computed on adata by evaluate().
    """
    # Read existing leiden results (computed by evaluate())
    y = adata.obs[label_key].values
    pred = adata.obs['leiden'].values
    nmi = normalized_mutual_info_score(y, pred)
    ari = adjusted_rand_score(y, pred)
    n_clusters = adata.obs['leiden'].nunique()

    # Subsample for visualization
    adata_sub = sc.pp.subsample(adata, fraction=0.3, random_state=seed, copy=True)
    sc.tl.umap(adata_sub, random_state=seed)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get unique categories and create color maps
    cell_cats = sorted(adata_sub.obs[label_key].unique())
    batch_cats = sorted(adata_sub.obs[batch_key].unique())
    cluster_cats = sorted(adata_sub.obs['leiden'].unique(), key=lambda x: int(x))

    cell_palette = sc.pl.palettes.default_20 if len(cell_cats) <= 20 else sc.pl.palettes.default_102
    batch_palette = sc.pl.palettes.default_20 if len(batch_cats) <= 20 else sc.pl.palettes.default_102
    cluster_palette = sc.pl.palettes.default_20 if len(cluster_cats) <= 20 else sc.pl.palettes.default_102

    cell_colors = {cat: cell_palette[i] for i, cat in enumerate(cell_cats)}
    batch_colors = {cat: batch_palette[i] for i, cat in enumerate(batch_cats)}
    cluster_colors = {cat: cluster_palette[i] for i, cat in enumerate(cluster_cats)}

    # Plot celltype
    sc.pl.umap(adata_sub, color=label_key, ax=axes[0], show=False, legend_loc=None,
               frameon=False, title='Cell Type', palette=[cell_colors[c] for c in cell_cats])

    # Plot batch
    sc.pl.umap(adata_sub, color=batch_key, ax=axes[1], show=False, legend_loc=None,
               frameon=False, title='Batch', palette=[batch_colors[c] for c in batch_cats])

    # Plot leiden
    sc.pl.umap(adata_sub, color='leiden', ax=axes[2], show=False, legend_loc=None,
               frameon=False, title=f'Leiden (n={n_clusters}, NMI={nmi:.3f}, ARI={ari:.3f})',
               palette=[cluster_colors[c] for c in cluster_cats])

    # Add celltype legend at bottom
    cell_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                           markerfacecolor=cell_colors[cat], markersize=6) for cat in cell_cats]
    axes[0].legend(handles=cell_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(cell_cats), 4), fontsize=7, frameon=False)

    # Add batch legend at bottom
    batch_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                            markerfacecolor=batch_colors[cat], markersize=6) for cat in batch_cats]
    axes[1].legend(handles=batch_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(batch_cats), 6), fontsize=7, frameon=False)

    # Add cluster legend at bottom
    cluster_handles = [Line2D([0], [0], marker='o', color='w', label=f'C{cat}',
                              markerfacecolor=cluster_colors[cat], markersize=6) for cat in cluster_cats]
    axes[2].legend(handles=cluster_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(cluster_cats), 6), fontsize=7, frameon=False)

    plt.subplots_adjust(bottom=0.2, wspace=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def best_leiden_by_nmi(adata: sc.AnnData, label_key: str, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0), random_state=42):
    best = (-1.0, -1.0, None)
    y = adata.obs[label_key].values
    for r in resolutions:
        sc.tl.leiden(adata, key_added='cluster', resolution=r, random_state=random_state)
        pred = adata.obs['cluster'].values
        nmi = normalized_mutual_info_score(y, pred)
        ari = adjusted_rand_score(y, pred)
        if nmi > best[0]:
            best = (nmi, ari, r)
    return best


def seq_preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Standard preprocessing for scRNA-seq data."""
    adata = adata.copy()

    if issparse(adata.X):
        adata.X = adata.X.toarray()

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', subset=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def load_adata(path: str, data_type: str, preprocess: bool = False, batch_key: str = 'BATCH', cell_type_key: str = 'celltype'):
    """Load AnnData and extract data, batch_labels, cell_types."""
    adata = sc.read_h5ad(path)

    if data_type == 'seq' and preprocess:
        adata = seq_preprocess(adata)

    data = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    # Numeric codes for training
    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    cell_types = pd.Categorical(adata.obs[cell_type_key]).codes if cell_type_key in adata.obs.columns else None

    # Original string labels for visualization
    batch_names = adata.obs[batch_key].values
    cell_type_names = adata.obs[cell_type_key].values if cell_type_key in adata.obs.columns else None

    return data, batch_labels, cell_types, batch_names, cell_type_names


def evaluate(
    adata,
    adata_raw,
    embed='X_biobatchnet',
    batch_key='BATCH',
    label_key='celltype',
    fraction=1.0,
):
    # Compute neighbors + leiden on FULL adata (results persist for visualization)
    sc.pp.neighbors(adata, use_rep=embed, random_state=42)
    nmi, ari, best_r = best_leiden_by_nmi(adata, label_key, random_state=42)
    sc.tl.leiden(adata, key_added='leiden', resolution=best_r, random_state=42)
    adata.uns['leiden_resolution'] = best_r

    # Subsample for expensive scib batch metrics
    adata_sub = sc.pp.subsample(adata, fraction=fraction, random_state=42, copy=True)
    adata_raw_sub = adata_raw[adata_sub.obs_names].copy()

    sc.pp.neighbors(adata_sub, use_rep=embed, random_state=42)
    sc.pp.pca(adata_raw_sub)
    sc.pp.neighbors(adata_raw_sub, use_rep='X_pca')

    # Batch correction metrics
    ilisi = scib.metrics.ilisi_graph(adata_sub, batch_key=batch_key, type_='embed', use_rep=embed)
    graph_conn = scib.me.graph_connectivity(adata_sub, label_key=label_key)
    asw_batch = scib.me.silhouette_batch(adata_sub, batch_key=batch_key, label_key=label_key, embed=embed)
    pcr = scib.me.pcr_comparison(adata_raw_sub, adata_sub, covariate=batch_key, embed=embed)

    # Biological conservation metrics (leiden NMI/ARI from full data above)
    asw_cell = scib.me.silhouette(adata_sub, label_key=label_key, embed=embed)

    # Aggregate scores
    batch_score = (ilisi + graph_conn + asw_batch + pcr) / 4
    bio_score = (asw_cell + ari + nmi) / 3
    total_score = (batch_score + bio_score) / 2

    return {
        'iLISI': ilisi,
        'GraphConn': graph_conn,
        'ASW_batch': asw_batch,
        'PCR': pcr,
        'BatchScore': batch_score,
        'ASW': asw_cell,
        'ARI': ari,
        'NMI': nmi,
        'BioScore': bio_score,
        'TotalScore': total_score,
    }


def independence_metrics(bio_z: np.ndarray, batch_z: np.ndarray) -> dict:
    corrs = []
    for i in range(bio_z.shape[1]):
        for j in range(batch_z.shape[1]):
            r, _ = pearsonr(bio_z[:, i], batch_z[:, j])
            corrs.append(abs(r))
    mean_corr = np.mean(corrs)
    max_corr = np.max(corrs)

    # Mutual information: estimate MI between batch_z and each dimension of bio_z
    mi_scores = []
    for i in range(bio_z.shape[1]):
        mi = mutual_info_regression(batch_z, bio_z[:, i], random_state=42)
        mi_scores.append(mi.mean())
    mean_mi = np.mean(mi_scores)

    return {
        'mean_abs_corr': mean_corr,
        'max_abs_corr': max_corr,
        'mean_MI': mean_mi,
    }


def _probe_balacc(z, labels, n_splits=5, seed=42):
    """Linear probe: z -> labels. Returns balanced accuracy and chance level."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    X = StandardScaler().fit_transform(z)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    pred = cross_val_predict(clf, X, y, cv=cv)
    balacc = float(balanced_accuracy_score(y, pred))
    chance = float(1.0 / n_classes) if n_classes > 0 else 0.0
    return balacc, chance


def _latent_r2(src, tgt, n_splits=5, seed=42):
    """CV R² (Ridge): src -> tgt, averaged over target dimensions."""
    X = StandardScaler().fit_transform(src)
    Y = StandardScaler().fit_transform(tgt)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ridge = Ridge(alpha=1.0, random_state=seed)
    r2s = []
    for j in range(Y.shape[1]):
        r2s.append(r2_score(Y[:, j], cross_val_predict(ridge, X, Y[:, j], cv=cv)))
    return float(np.mean(r2s))


def independence_eval(bio_z, batch_z, batch_labels, cell_labels=None, n_splits=5, seed=42):
    """Independence evaluation between bio and batch latent spaces.

    Returns dict with probe balanced-accuracy scores and cross-validated R²
    measuring leakage between the two latent spaces.
    """
    results = {}

    # Probes: predict batch from bio (should be near chance) and from batch (sanity)
    balacc, chance = _probe_balacc(bio_z, batch_labels, n_splits, seed)
    results['batch_from_bio_balacc'] = balacc
    results['batch_from_bio_chance'] = chance
    results['batch_from_bio_gap'] = balacc - chance

    balacc, chance = _probe_balacc(batch_z, batch_labels, n_splits, seed)
    results['batch_from_batch_balacc'] = balacc
    results['batch_from_batch_chance'] = chance
    results['batch_from_batch_gap'] = balacc - chance

    if cell_labels is not None:
        balacc, chance = _probe_balacc(bio_z, cell_labels, n_splits, seed)
        results['celltype_from_bio_balacc'] = balacc
        results['celltype_from_bio_gap'] = balacc - chance

        balacc, chance = _probe_balacc(batch_z, cell_labels, n_splits, seed)
        results['celltype_from_batch_balacc'] = balacc
        results['celltype_from_batch_gap'] = balacc - chance

    # CV R²: batch <-> bio
    results['r2_batch_to_bio'] = _latent_r2(batch_z, bio_z, n_splits, seed)
    results['r2_bio_to_batch'] = _latent_r2(bio_z, batch_z, n_splits, seed)

    return results


def aggregate_seeds(run_dir: str, save: bool = True) -> pd.DataFrame:
    """Aggregate metrics across seeds, compute mean and std.

    Args:
        run_dir: Path to run directory containing seed_* subdirectories
        save: Whether to save summary.csv to run_dir

    Returns:
        DataFrame with mean and std for each metric
    """
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    all_metrics = []

    for seed_dir in sorted(run_dir.glob('seed_*')):
        csv_path = seed_dir / 'metrics.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        row = df.iloc[-1]  # last row
        metrics = {}

        # Parse eval metrics
        if 'eval' in row and pd.notna(row['eval']):
            eval_dict = json.loads(row['eval'])
            metrics.update(eval_dict)

        # Parse independence metrics
        if 'independence' in row and pd.notna(row['independence']):
            inde_dict = json.loads(row['independence'])
            metrics.update(inde_dict)

        metrics['seed'] = seed_dir.name
        all_metrics.append(metrics)

    if not all_metrics:
        print(f"No metrics found in {run_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)

    # Compute mean and std for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'std': df[numeric_cols].std(),
    })

    if save:
        summary_path = run_dir / 'summary.csv'
        summary.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")

        # Also print formatted results
        print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10}")
        print("-" * 37)
        for metric in summary.index:
            print(f"{metric:<15} {summary.loc[metric, 'mean']:>10.4f} {summary.loc[metric, 'std']:>10.4f}")

    return summary
