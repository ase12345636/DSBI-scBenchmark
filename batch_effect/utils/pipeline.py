import os
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.io import mmwrite
from scipy.sparse import issparse, csc_matrix

from utils.dataset import preprocess_adata
from utils.evaluation import evaluate_embedding_scib
from utils.function import find_best_leiden_resolution, MemoryTracker

# Default preprocessing: normalize + log + per-batch HVG (2000 genes).
# Used by tools whose published protocol expects log-normalized HVG input
# (Harmony / scDML / scCobra). Tools with different requirements (scVI,
# Scanorama, Seurat) override this via the `preprocess_fn` kwarg of
# `run_integration_pipeline`.


# ============================================================
# Argument Parsing
# ============================================================

def create_argument_parser(tool_name):
    """Create the standard argument parser shared by all batch correction tools."""
    parser = argparse.ArgumentParser(description=f'Run {tool_name} batch correction')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to input h5ad file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--batch_key', type=str, default='batch', help='Batch column name')
    parser.add_argument('--celltype_key', type=str, default='celltype', help='Cell type column name')
    parser.add_argument('--run_times', type=int, default=1, help='Number of times to run')
    parser.add_argument('--compute_oc', action='store_true', help='Compute OverCorrection score (slow)')
    parser.add_argument('--ATAC', action='store_true', help='ATAC mode: skip preprocessing')
    return parser


# ============================================================
# Data Loading & Preprocessing
# ============================================================

def load_dataset(dataset_path):
    """Load an h5ad dataset from disk."""
    print(f"Loading data from {dataset_path}")
    return sc.read_h5ad(dataset_path)


def preprocess_or_copy(adata_raw, celltype_key, batch_key, compute_oc, atac,
                       preprocess_fn=None):
    """Preprocess data or copy raw (for ATAC / OC modes that skip preprocessing).

    Parameters
    ----------
    preprocess_fn : callable or None
        Optional preprocessing function with signature
        ``fn(adata, label_col, batch_col) -> AnnData``.
        Defaults to ``preprocess_adata`` (log + HVG-2000).
    """
    if compute_oc or atac:
        return adata_raw.copy()
    fn = preprocess_fn if preprocess_fn is not None else preprocess_adata
    return fn(adata_raw.copy(), celltype_key, batch_key)


def generate_seed():
    """Generate a time-based random seed and set numpy's global seed."""
    seed = int(time.time() * 1000000) % (2**31)
    np.random.seed(seed)
    return seed


# ============================================================
# Post-Integration Utilities
# ============================================================

def compute_umap_and_save(adata, use_rep, seed, batch_key, celltype_key, save_path):
    """Compute neighbors + UMAP, then save UMAP coordinates to CSV."""
    sc.pp.neighbors(adata, use_rep=use_rep, random_state=seed)
    sc.tl.umap(adata)
    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
    umap_df['batch'] = adata.obs[batch_key].values
    umap_df['celltype'] = adata.obs[celltype_key].values
    umap_df.to_csv(os.path.join(save_path, "umap_coordinates.csv"), index=False)


def plot_umaps(adata, batch_key, celltype_key, save_path, tool_name):
    """Save batch / celltype / leiden UMAP plots."""
    for color_key, suffix in [(batch_key, 'batch'), (celltype_key, 'celltype'), ('leiden', 'leiden')]:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(adata, color=[color_key], save=None, show=False,
                    frameon=False, title='', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.savefig(os.path.join(save_path, f"{tool_name}_umap_{suffix}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_and_save_metrics(adata, embed_key, batch_key, celltype_key,
                               best_resolution, seed, runtime, mem_metrics,
                               save_path, tool_name, compute_oc=False,
                               precomputed_cluster_key=None):
    """Evaluate the embedding, save metrics CSV, and print a summary."""
    metrics = evaluate_embedding_scib(
        adata, embed_key=embed_key, batch_key=batch_key, celltype_key=celltype_key,
        leiden_resolution=best_resolution,
        random_state=seed,
        compute_oc=compute_oc,
        precomputed_cluster_key=precomputed_cluster_key,
    )
    metrics['runtime_seconds'] = runtime
    metrics['best_leiden_resolution'] = best_resolution
    metrics['peak_ram_mb'] = mem_metrics['peak_ram_mb']
    metrics['peak_vram_mb'] = mem_metrics['peak_vram_mb']

    pd.DataFrame([metrics]).to_csv(
        os.path.join(save_path, f"{tool_name}_metrics.csv"), index=False)

    print(f"\n[Seed {seed}] {tool_name} metrics (with best resolution {best_resolution:.1f}):")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  RAM: {mem_metrics['peak_ram_mb']:.2f} MB, VRAM: {mem_metrics['peak_vram_mb']:.2f} MB")
    print(f"  NMI={metrics['NMI']:.4f}, ARI={metrics['ARI']:.4f}")
    print(f"  ASW_bio={metrics['ASW_bio']:.4f}, ASW_batch={metrics['ASW_batch']:.4f}")
    print(f"  AVG_bio={metrics['AVG_bio']:.4f}, AVG_batch={metrics['AVG_batch']:.4f}")

    return metrics


# ============================================================
# R Interop (Python ↔ R data exchange)
# ============================================================

def export_adata_for_r(adata, output_dir):
    """Export adata as mtx + csv for native R loading (no reticulate needed).

    Creates: matrix.mtx (genes × cells), barcodes.csv, features.csv, metadata.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # Counts matrix — transpose to genes × cells (R convention)
    X = adata.X
    if not issparse(X):
        X = csc_matrix(X)
    mmwrite(os.path.join(output_dir, "matrix.mtx"), X.T)

    pd.DataFrame({'barcode': adata.obs_names}).to_csv(
        os.path.join(output_dir, "barcodes.csv"), index=False)
    pd.DataFrame({'feature': adata.var_names}).to_csv(
        os.path.join(output_dir, "features.csv"), index=False)
    adata.obs.to_csv(os.path.join(output_dir, "metadata.csv"))


def import_embedding_from_r(adata, csv_path, embed_key):
    """Read an embedding CSV exported by R and attach to adata.obsm."""
    emb_df = pd.read_csv(csv_path, index_col=0)
    # Align by cell names in case R changed cell order
    emb_df = emb_df.reindex(adata.obs_names)
    adata.obsm[embed_key] = emb_df.values


# ============================================================
# Main Pipeline Orchestrator
# ============================================================

def cleanup_r_exchange(r_exchange_dir):
    """Remove the shared R exchange directory after all runs complete."""
    if r_exchange_dir and os.path.exists(r_exchange_dir):
        import shutil
        shutil.rmtree(r_exchange_dir)
        print(f"Cleaned up R exchange directory: {r_exchange_dir}")


def run_integration_pipeline(tool_name, integrate_fn, embed_key, args,
                             run_pca=True, r_exchange_dir=None,
                             preprocess_fn=None):
    """
    Run the full integration-evaluation pipeline for a batch correction tool.

    Parameters
    ----------
    tool_name : str
        Name of the tool (used for directory and file naming).
    integrate_fn : callable
        ``integrate_fn(adata, seed, run_out_path, args) -> adata``
        Must store the integrated embedding in ``adata.obsm[embed_key]``.
    embed_key : str
        Key in ``adata.obsm`` where the integrated embedding lives.
    args : argparse.Namespace
        Parsed CLI arguments (from ``create_argument_parser``).
    run_pca : bool
        If True, compute PCA before timing starts (standard for most tools).
        If False, ``integrate_fn`` is responsible for computing PCA (e.g. raw).
    r_exchange_dir : str or None
        Path to shared R exchange directory (for R-based tools).
        If provided, exported once before runs and cleaned up after all runs.
    """
    os.makedirs(args.save_path, exist_ok=True)
    adata_raw = load_dataset(args.dataset_path)

    for run_id in range(1, args.run_times + 1):
        print(f"\n{'='*60}")
        print(f"Run {run_id}/{args.run_times}")
        print(f"{'='*60}\n")

        seed = generate_seed()
        run_out_path = os.path.join(args.save_path, f"{tool_name}/{run_id}/")
        os.makedirs(run_out_path, exist_ok=True)

        adata = preprocess_or_copy(adata_raw, args.celltype_key, args.batch_key,
                                    args.compute_oc, args.ATAC,
                                    preprocess_fn=preprocess_fn)
        print(f"Preprocessed dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")

        adata.obs_names_make_unique()

        if run_pca:
            sc.tl.pca(adata, random_state=seed)

        # Preserve raw PCA so it survives integration methods that overwrite / lose it
        raw_pca = adata.obsm.get('X_pca', None)
        pre_obs_names = adata.obs_names.copy()
        if raw_pca is not None:
            raw_pca = raw_pca.copy()

        print(f"\n{'='*60}")
        print(f"Running {tool_name} batch correction...")
        print(f"{'='*60}\n")

        # --- Timed integration ---
        tracker = MemoryTracker()
        tracker.start()
        start_time = time.time()
        result = integrate_fn(adata, seed, run_out_path, args)
        end_time = time.time()
        mem_metrics = tracker.stop()

        # integrate_fn may return (adata, algo_runtime) to override wall-clock time
        # (e.g. Seurat reports R-side algorithm time, excluding I/O overhead)
        if isinstance(result, tuple):
            adata, runtime = result
        else:
            adata = result
            runtime = end_time - start_time

        # Restore raw PCA if it was lost (e.g. after sc.concat in Scanorama)
        # Use obs_name-based alignment in case integration changed cell order (e.g. Seurat)
        if raw_pca is not None:
            if list(adata.obs_names) == list(pre_obs_names):
                adata.obsm['X_pca'] = raw_pca
            else:
                raw_pca_df = pd.DataFrame(raw_pca, index=pre_obs_names)
                adata.obsm['X_pca'] = raw_pca_df.reindex(adata.obs_names).values

        # --- Post-integration evaluation ---
        compute_umap_and_save(adata, embed_key, seed,
                              args.batch_key, args.celltype_key, run_out_path)

        leiden_results = find_best_leiden_resolution(
            adata, celltype_key=args.celltype_key, seed=seed,
            res_min=0.1, res_max=2.0, res_step=0.1,
            save_path=os.path.join(run_out_path, "resolution_search.csv")
        )

        plot_umaps(adata, args.batch_key, args.celltype_key, run_out_path, tool_name)

        evaluate_and_save_metrics(
            adata, embed_key, args.batch_key, args.celltype_key,
            leiden_results['best_resolution'], seed, runtime, mem_metrics,
            run_out_path,
            tool_name,
            compute_oc=args.compute_oc,
            precomputed_cluster_key='leiden',
        )
        print(f"Results saved to {run_out_path}")

    cleanup_r_exchange(r_exchange_dir)

    print(f"\n{'='*60}")
    print(f"All runs completed!")
    print(f"Results saved to {args.save_path}")
    print(f"{'='*60}")
