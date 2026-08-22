# Cell type annotation benchmark

Reference-based cell type annotation benchmark implemented in the same style
as `batch_effect`: every tool has a small wrapper, while data loading,
repeated runs, metrics, UMAP and result writing are handled by shared modules.

The benchmark includes CellTypist, CHETAH, Seurat label transfer, SingleR and
scGPT. Each tool uses an independent Conda environment to avoid Python/R and
package-version conflicts.

## Common input

Both inputs are AnnData (`.h5ad`) files. The four metadata arguments are
independent because reference and query datasets may use different column
names:

- `--train_celltype`: reference labels used to train the classifier.
- `--test_celltype`: query ground truth used only for evaluation.
- `--train_batch`: reference batch column; currently recorded/validated only.
- `--test_batch`: query batch column used in saved coordinates and batch UMAP.

Pass the literal value `None` when a column does not exist. With
`--test_celltype None`, prediction and UMAP outputs are still generated but
classification metrics, confusion matrices and ground-truth UMAPs are
skipped. All included tools are supervised, so `--train_celltype None` is
accepted as an explicit missing value but cannot be executed.

## Common Python packages

Every method environment needs the common Python evaluation/plotting stack:

```bash
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
```

## CellTypist

Create a standalone environment:

```bash
conda create -n Celltypist -c conda-forge python=3.10 pip -y
conda activate Celltypist
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
pip install celltypist==1.7.1
```

## CHETAH

CHETAH is isolated from Seurat and SingleR because all three have independent
R/Bioconductor dependency trees:

```bash
conda create -n CHETAH -c conda-forge \
  python=3.10 pip r-base=4.3 r-argparse r-matrix r-biocmanager -y
conda activate CHETAH
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
Rscript -e 'BiocManager::install(c("SingleCellExperiment", "CHETAH"), ask=FALSE, update=FALSE)'
```

## Seurat

Use Conda's compiled R packages so Seurat does not share libraries with the
Bioconductor methods:

```bash
conda create -n Seurat -c conda-forge \
  python=3.10 pip r-base=4.3 r-argparse r-matrix \
  r-seurat r-seuratobject -y
conda activate Seurat
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
```

## SingleR

```bash
conda create -n SingleR -c conda-forge -c bioconda \
  python=3.10 pip r-base=4.3 r-argparse r-matrix r-biocmanager \
  bioconductor-singler bioconductor-singlecellexperiment \
  bioconductor-scran -y
conda activate SingleR
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
```

## scGPT

The versions below keep PyTorch and torchtext matched. Flash Attention is
optional and deliberately omitted to avoid CUDA/build conflicts.

```bash
conda create -n scGPT -c conda-forge python=3.10 pip -y
conda activate scGPT
pip install torch==2.1.2 torchvision==0.16.2 \
  torchaudio==2.1.2 torchtext==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
pip install scgpt==0.2.4 gdown
```

This benchmark uses the official `whole-human` checkpoint recommended by the
scGPT project. Download it once with:

```bash
conda activate scGPT
python scGPT/download_checkpoint.py \
  --output_dir scGPT/models/scGPT_human
```

The directory is valid only when it contains all three files:

```text
scGPT/models/scGPT_human/
├── args.json
├── best_model.pt
└── vocab.json
```

`run_script.sh` performs the same check and downloads the checkpoint
automatically when these files are absent. A different checkpoint can be
passed with `--scgpt_model_dir`.

## Run all methods

```bash
bash run_script.sh \
    --train_path reference.h5ad \
    --test_path query.h5ad \
    --save_path results/ \
    --train_celltype reference_label \
    --test_celltype query_ground_truth \
    --train_batch reference_batch \
    --test_batch query_batch \
    --n_runs 5 \
    --base_seed 42
```

For a dataset without query batch or ground-truth metadata:

```bash
bash run_script.sh \
    --train_path reference.h5ad \
    --test_path unlabelled_query.h5ad \
    --save_path results/ \
    --train_celltype reference_label \
    --test_celltype None \
    --train_batch None \
    --test_batch None \
    --n_runs 1
```

Do not use the same cells for reference and query when reporting benchmark
performance; doing so measures training-set performance.

## Output

Each run is saved to `SAVE_PATH/METHOD/RUN_ID/`:

- `METHOD_metrics.csv`
- `per_class_metrics.csv`
- `predictions.csv`
- `confusion_matrix.csv` and `METHOD_confusion_matrix.png`
- `umap_coordinates.csv`
- prediction, ground-truth, batch and comparison UMAP PNGs when their keys are
  available

Metrics include accuracy, balanced accuracy, macro/micro/weighted precision,
recall and F1, one-vs-rest AUC, MCC, Cohen's kappa, runtime, RAM and VRAM.
Each method directory also contains `metrics_summary.csv` and
`metrics_aggregate.csv` across repeated runs.

The optional `utils/analysis.py` can aggregate already completed method runs,
but it is not called by `run_cmd.sh` or `run_script.sh`.
