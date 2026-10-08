"""
Create perturbed immune datasets for overcorrection evaluation.

For each target cell type (CD4+ T cells, CD14+ Monocytes):
  1. Load and preprocess the immune dataset
  2. Copy the target cell type's cells
  3. Add +1 perturbation to the copied cells' expression
  4. Label the copies as "<celltype>(perturbed)"
  5. Draw UMAP (colored by batch and celltype)
  6. Save the combined dataset as .h5ad
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ---- Configuration ----
DATASET_PATH = "/Group16T/raw_data/scCobra/Immune_ALL_human.h5ad"
SAVE_DIR = "/Group16T/common/ccuc/data"
BATCH_KEY = "batch"
CELLTYPE_KEY = "final_annotation"

TARGETS = [
    {
        "celltype": "CD4+ T cells",
        "perturbed_name": "CD4+ T cells(perturbed)",
        "output_filename": "Immune_perturbed_CD4T.h5ad",
        "umap_prefix": "CD4T",
    },
    {
        "celltype": "CD14+ Monocytes",
        "perturbed_name": "CD14+ Monocytes(perturbed)",
        "output_filename": "Immune_perturbed_CD14.h5ad",
        "umap_prefix": "CD14",
    },
]

os.makedirs(SAVE_DIR, exist_ok=True)


def preprocess_perturbation_input(
    adata,
    batch_col,
    min_genes=200,
    min_cells=3,
    mt_threshold=5.0,
    target_sum=1e4,
    n_top_genes=2000,
):
    """Local preprocessing used only to construct perturbation datasets."""
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs["pct_counts_mt"] < mt_threshold].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat",
            batch_key=batch_col,
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
    return adata

# ---- Load raw data ----
print(f"Loading data from {DATASET_PATH}")
adata_raw = sc.read_h5ad(DATASET_PATH)
print(f"Raw dataset: {adata_raw.shape[0]} cells, {adata_raw.shape[1]} genes")

for target in TARGETS:
    celltype = target["celltype"]
    perturbed_name = target["perturbed_name"]
    output_path = os.path.join(SAVE_DIR, target["output_filename"])
    umap_prefix = target["umap_prefix"]

    print(f"\n{'='*60}")
    print(f"Processing: {celltype}")
    print(f"{'='*60}")

    # ---- Preprocess ----
    print("Preprocessing...")
    adata = preprocess_perturbation_input(adata_raw, batch_col=BATCH_KEY)
    print(f"Preprocessed: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # ---- Copy target cells and perturb ----
    mask = adata.obs[CELLTYPE_KEY] == celltype
    n_target = mask.sum()
    print(f"Found {n_target} '{celltype}' cells")

    if n_target == 0:
        print(f"  WARNING: No cells of type '{celltype}' found, skipping.")
        continue

    adata_target = adata[mask].copy()

    # Perturbation: +1 to expression matrix
    if sp.issparse(adata_target.X):
        adata_target.X = adata_target.X.toarray() + 1.0
        adata_target.X = sp.csr_matrix(adata_target.X)
    else:
        adata_target.X = adata_target.X + 1.0

    # Rename celltype to perturbed name
    adata_target.obs[CELLTYPE_KEY] = perturbed_name

    # Make obs_names unique so they don't clash with original
    adata_target.obs_names = [f"{name}_perturbed" for name in adata_target.obs_names]

    # ---- Concatenate original + perturbed ----
    adata_combined = sc.concat([adata, adata_target], join="inner")
    adata_combined.obs_names_make_unique()

    # Rename celltype column to 'celltype' for consistency with downstream pipeline
    adata_combined.obs["celltype"] = adata_combined.obs[CELLTYPE_KEY].astype(str)
    if CELLTYPE_KEY != "celltype":
        adata_combined.obs.drop(columns=[CELLTYPE_KEY], inplace=True, errors="ignore")

    print(f"Combined dataset: {adata_combined.shape[0]} cells, {adata_combined.shape[1]} genes")
    celltypes = adata_combined.obs["celltype"].value_counts()
    print("Cell type distribution:")
    for ct, count in celltypes.items():
        print(f"  {ct}: {count}")

    # ---- PCA + UMAP ----
    print("Computing PCA and UMAP...")
    sc.tl.pca(adata_combined, n_comps=50)
    sc.pp.neighbors(adata_combined)
    sc.tl.umap(adata_combined)

    # ---- Plot UMAPs ----
    # Batch UMAP
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(
        adata_combined,
        color=BATCH_KEY,
        ax=ax,
        show=False,
        frameon=False,
        title=f"Immune + {perturbed_name} — Batch",
        s=5,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    batch_umap_path = os.path.join(SAVE_DIR, f"{umap_prefix}_perturbed_umap_batch.png")
    plt.savefig(batch_umap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved batch UMAP: {batch_umap_path}")

    # Celltype UMAP
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(
        adata_combined,
        color="celltype",
        ax=ax,
        show=False,
        frameon=False,
        title=f"Immune + {perturbed_name} — Cell Type",
        s=5,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    celltype_umap_path = os.path.join(SAVE_DIR, f"{umap_prefix}_perturbed_umap_celltype.png")
    plt.savefig(celltype_umap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved celltype UMAP: {celltype_umap_path}")

    # ---- Save h5ad ----
    adata_combined.write(output_path)
    print(f"Saved dataset: {output_path}")

print(f"\n{'='*60}")
print("All done!")
print(f"{'='*60}")
