# Cell type annotation benchmark

Reference-based cell type annotation benchmark implemented in the same style
as `batch_effect`: every tool has a small wrapper, while data loading,
repeated runs, metrics, UMAP and result writing are handled by the project-level
[`utils`](utils) package mirrored with scLightGAT.

The benchmark includes CellTypist, CHETAH, Seurat label transfer, SingleR,
scANVI, scGraphformer and scGPT. Each tool uses a method-appropriate Conda
environment to avoid Python/R and package-version conflicts.

## Common input

`--train_path` is always one AnnData (`.h5ad`) reference file. In standard
mode, `--test_path` may be one query file or a directory; every directly
contained `.h5ad` file is then predicted from the same model for that run and
seed. Metadata arguments are
independent because reference and query datasets may use different column
names:

- `--train_celltype`: reference labels used to train the classifier.
- `--test_celltype`: query ground truth used only for evaluation.
- `--train_batch`: reference batch column, available to methods with native
  batch-aware processing.
- `--test_batch`: query batch column used in saved coordinates and batch UMAP.
- `--test_batch_map`: optional `DATASET=COLUMN,...` overrides for a standard
  query directory whose files use different batch-column names.
Pass the literal value `None` when a column does not exist. With
`--test_celltype None`, prediction and UMAP outputs are still generated but
classification metrics, confusion matrices and ground-truth UMAPs are
skipped. All included tools are supervised, so `--train_celltype None` is
accepted as an explicit missing value but cannot be executed.

## Common Python packages

The existing methods and scANVI use the common Python evaluation/plotting
stack below. scGraphformer uses its separately pinned CUDA-13 environment
documented later because its verified stack uses NumPy 2:

```bash
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
```

### Verified tool versions

The current benchmark implementation was reviewed against these installed
versions. This records the verified environment only; it does not install,
upgrade or lock packages:

| Package | Verified version |
| --- | --- |
| CellTypist | 1.7.1 |
| scikit-learn | 1.7.2 |
| Seurat | 5.3.0 |
| SeuratObject | 5.2.0 |
| CHETAH | 1.18.0 |
| SingleR | 2.8.0 |
| scGPT | 0.2.4 |
| scvi-tools (scANVI) | 1.3.3 |
| PyTorch (scANVI environment) | 2.12.0a0+git29094e4, CUDA 13.0 |
| scGraphformer source reference | commit `75c76fc6b66195297a764d6e4569b3c01e084538` |
| PyTorch (scGraphformer environment) | 2.11.0+cu130 |

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

Create CHETAH from scratch in its own environment. Remove any old `CHETAH`
environment first; all R and Bioconductor dependencies are then installed by
Conda/Bioconda in one transaction:

```bash
conda create -n CHETAH \
  --copy --override-channels --strict-channel-priority \
  -c conda-forge -c bioconda \
  python=3.10 pip \
  r-base=4.3 r-argparse r-matrix \
  r-data.table r-htmlwidgets r-plotly r-shiny \
  bioconductor-data-packages=20260207=hdfd78af_0 \
  bioconductor-genomeinfodbdata=1.2.11=r43hdfd78af_0 \
  bioconductor-chetah=1.18 \
  bioconductor-singlecellexperiment=1.24 \
  bioconductor-summarizedexperiment=1.32 \
  bioconductor-genomicranges=1.54 \
  bioconductor-delayedarray=0.28 \
  -y

conda activate CHETAH
python -m pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
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
conda create -n SingleR \
  -c conda-forge -c bioconda \
  --strict-channel-priority \
  python=3.10 \
  r-base=4.4 \
  r-argparse \
  r-matrix \
  r-biocmanager \
  bioconductor-singler=2.8.0 \
  bioconductor-singlecellexperiment=1.28.0 \
  bioconductor-scran=1.34.0 \
  -y
conda activate SingleR
pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn
```

## scGPT

On GB10, keep the platform/vendor CUDA 13 AArch64 PyTorch build that supports
the GPU's `sm_121` architecture. Do not replace it with the older CUDA 12.1
(`cu121`) wheels. Follow the current NVIDIA GB10 image guidance or PyTorch
platform guidance for the base Torch installation; an exact nightly wheel is
intentionally not pinned here because those builds expire and change.

torchtext ABI compatibility is not a prerequisite for this benchmark. The
preserved scGPT code installs a narrow pure-Python `Vocab` compatibility shim
when torchtext's compiled extension cannot load with the platform Torch build.
Flash Attention is deliberately omitted because available external packages do
not implement the legacy API expected by scGPT 0.2.4 on this stack.

Do not add a bare `pip install flash_attn`. The reported
`ModuleNotFoundError: No module named 'torch'` comes from pip build isolation;
`--no-build-isolation` only lets the build see the existing Torch and does not
solve the remaining API, CUDA, or GPU-architecture compatibility.

scGPT 0.2.4 imports the legacy `flash_attn.flash_attention.FlashMHA` API,
whereas current FlashAttention and the CUDA 13/Blackwell implementation expose
newer APIs. The model is constructed with the tutorial settings
`use_fast_transformer=True` and `fast_transformer_backend="flash"`. When the
legacy `FlashMHA` API is unavailable, scGPT 0.2.4 applies its official fallback
to the standard PyTorch encoder, which can still dispatch to native
scaled-dot-product attention kernels when Torch, CUDA and the GPU support them.
Both the requested and effective backends, encoder classes and native SDPA
flags are recorded in the scGPT model identity. No separate FlashAttention
package is required.

```bash
conda create -n scGPT -c conda-forge python=3.10 pip -y
conda activate scGPT
# Install or retain the GB10 vendor-supported CUDA 13 AArch64 PyTorch build.
python -m pip install "numpy<2" pandas scipy scikit-learn \
  anndata scanpy matplotlib seaborn umap-learn \
  scgpt==0.2.4 gdown ipython
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); assert torch.cuda.is_available(); assert str(torch.version.cuda).startswith("13."); assert "sm_121" in torch.cuda.get_arch_list()'
PYTHONPATH=. python -c 'import sys; sys.path.insert(0, "scGPT"); import model_impl; print("scGPT compatibility imports OK")'
```

Checkpoint loading follows the resolved backend: Flash retains `Wqkv.*`, while
the standard PyTorch fallback applies scGPT's official `Wqkv.*` to `in_proj_*`
remap. A strict audit verifies all 24 QKV tensors from the 12-layer whole-human
checkpoint by exact tensor equality; unknown or non-classifier shape mismatches
abort the run instead of silently leaving encoder weights uninitialized.

This benchmark uses the official `whole-human` checkpoint recommended by the
scGPT project. `run_script.sh` downloads it automatically when it is missing or
invalid. It can also be downloaded explicitly with:

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

Before running scGPT, `run_script.sh` validates the JSON files and the PyTorch
archive. A missing or invalid checkpoint is downloaded to a temporary sibling
directory, fully loaded for validation, and only then published to the model
directory. A different checkpoint can be passed with `--scgpt_model_dir`.

## scANVI

The scANVI adapter is verified with `scvi-tools==1.3.3` in the existing
`sctools` environment. It follows the official scArches reference-mapping
workflow: 2,000 reference-only HVGs, the recommended SCVI architecture for
query mapping, automatic SCVI epochs, 20 labelled SCANVI epochs with 100
samples per label, and 100 query-mapping epochs with zero weight decay.
These method settings are fixed internally and are not exposed as command-line
options; only the shared benchmark inputs and resource controls remain public.

The scANVI adapter asks the common loader to retain raw counts from
`layers/counts`, and scvi-tools registers that layer explicitly. The common
pipeline may load ground truth for later evaluation, but it removes that
column before invoking the adapter.
scANVI therefore receives only the query expression, indices, and explicitly
configured batch column. Every query mapping starts from the same immutable
reference checkpoint.

Verify the existing environment with:

```bash
conda activate sctools
python -c 'import scvi, torch; print(scvi.__version__, torch.__version__, torch.version.cuda)'
```

If `sctools` must be rebuilt on this GB10 host, first retain or install a
CUDA-13 PyTorch build supporting `sm_121` as described for scGPT, then install:

```bash
python -m pip install "numpy<2" pandas scipy scikit-learn \
  anndata==0.11.4 scanpy==1.9.8 matplotlib seaborn umap-learn \
  scvi-tools==1.3.3
```

## scGraphformer

Create the standalone environment with:

```bash
conda create --name scGraphformer --channel conda-forge \
  python=3.10 git pip numpy=2.0.2 pandas=2.2.3 scipy=1.14.1 \
  scikit-learn=1.6.0 h5py=3.12.1 matplotlib=3.9.3 \
  seaborn=0.13.2 'psutil>=6'
conda activate scGraphformer
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130'
python -m pip install anndata==0.11.1 scanpy==1.10.4 \
  torch-geometric==2.8.0.post1
mkdir -p "$CONDA_PREFIX/src"
git clone https://github.com/xyfan22/scGraphformer.git \
  "$CONDA_PREFIX/src/scGraphformer"
git -C "$CONDA_PREFIX/src/scGraphformer" checkout --detach \
  75c76fc6b66195297a764d6e4569b3c01e084538
```

On this workstation, a faster equivalent starting point is
`conda create --name scGraphformer --clone scLightGAT`, followed by the pinned
PyTorch, PyG and official-source installation commands above.

The wrapper loads and instantiates the model class directly from that pinned
official checkout; it does not contain a second implementation of the model.
The upstream PyTorch 1.13.1/CUDA 11.6 environment cannot target the GB10 GPU.
On this aarch64 host only, the wrapper supplies the two `torch_sparse`
operations used by the official class with native PyTorch `index_add_`.

The benchmark uses the authors' documented cross-platform command as a fixed
workflow: `train_prop=0.8`, 20 epochs, batch size 256, and graph, exact KNN and
HVG options enabled. Model and training settings are not exposed as
command-line options; only the shared benchmark inputs and resource controls
remain public.
For benchmark consistency, one reference model is trained per seed and reused
for all queries; query genes are aligned to the fixed reference gene order,
with missing genes filled with zero. This is the train-once/predict-many
adaptation required by the common benchmark runner, rather than the upstream
pair-specific retraining script.

## Run the benchmark launcher

The launcher initializes Conda from the path at the top of `run_script.sh`;
update that single path if the project is moved to another workstation.

```bash
bash run_script.sh \
    --train_path reference.h5ad \
    --test_path query.h5ad \
    --save_path results/ \
    --train_celltype reference_label \
    --test_celltype query_ground_truth \
    --train_batch reference_batch \
    --test_batch query_batch \
    --n_runs 5
```

For a query directory whose files use different batch-column names:

```bash
bash run_script.sh \
    --train_path reference.h5ad \
    --test_path independent_tests/ \
    --save_path results/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch None \
    --test_batch_map 'GSE115978=samples,GSE123139=Sample.Short.Name,Zhengsorted=None' \
    --n_runs 5
```

The launcher generates each run seed once from system entropy and stores it
under `SAVE_PATH/.benchmark_state/seeds/`. All methods in the same result root
reuse that persisted seed, including after a restart.

The retained CD8 T-cell scGraphformer runs were completed with a different
persisted seed schedule before the repositories were unified. A tool-specific
state records those five actual seeds so resume remains truthful and does not
rerun or relabel existing results. Do not treat CD8 scGraphformer run indices
as paired seeds with the other methods.

A directory run prepares the reference once for each run and seed, then predicts
every `.h5ad` file sequentially. CellTypist reuses one fitted classifier.
CHETAH builds its profiles/tree per classifier call, SingleR trains its
query-compatible index per query, and Seurat finds query-specific anchors (and
its implicit reference PCA) per query; these operations are included in each
query's prediction runtime. scANVI reuses one reference checkpoint and performs
an independent unlabeled scArches update for each query. scGraphformer reuses
one fixed reference model; when its optional graph path is enabled, it builds
a separate query KNN graph. Neither method sees query ground-truth labels.

Runtime uses one common wrapper boundary for every method. Reading the source
`.h5ad` files, evaluation and plotting are excluded; per-run method
preprocessing, training and prediction are included.
Python-to-R exchange and R subprocess startup are therefore included for the R
methods. scANVI query adaptation and scGraphformer query normalization and
forward inference are included in prediction runtime; optional graph
construction is also included when enabled. These
values represent end-to-end method execution, not isolated algorithm-kernel
time. Use fresh, uninterrupted runs for reportable timing: scGPT can resume an
interrupted epoch checkpoint, but the final metric cannot recover wall time
spent by the earlier terminated process.

Standard-mode automatic resume is always enabled; there is no resume flag.
Completion markers validate the seed, training options, input-file identity
and all required result files. Re-running the same command skips completed
datasets. R reference objects and scGPT checkpoints can be loaded when later
queries remain. scANVI, scGraphformer and CellTypist retrain if an incomplete
run still needs prediction. scGPT resumes at the next complete epoch if
fine-tuning was interrupted. Results created before these markers existed are
intentionally recomputed once, because the pipeline cannot prove that they came
from the same protocol.

The repository's `run_cmd.sh` runs the broad test directory, keeps CAF
separate because it uses a different training reference, and loops over the
five Subtype groups. The unified `run_script.sh` includes all seven benchmark
methods, including scANVI and scGraphformer.

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

## Defaults, fairness, and memory safeguards

No common wrapper samples, caps, or aggregates away training/reference cells.
Every method receives the full reference population. scGraphformer then uses a
random 80% reference training subset, matching the authors' cross-platform
configuration.
Query labels are never used to tune a method or its resource settings.

The formal launcher fixes `--n_jobs 12`. CellTypist receives `n_jobs=12`, and
SingleR receives both `MulticoreParam(workers=12)` and `num.threads=12`.
Seurat and CHETAH retain their installed algorithms while their BLAS/OpenMP
libraries are capped at twelve threads. scGPT DataLoaders use twelve workers. This
is one common resource ceiling, not dataset- or method-specific tuning.

| Method | Execution policy |
| --- | --- |
| CellTypist | Uses the official centered logistic-regression defaults, with only `random_state` and the common `n_jobs` allowance supplied. Prediction is run in one call unless its estimated dense scaling workspace would exceed 23 GiB; only then is it split into output-equivalent cell batches. Training is not batched. |
| CHETAH | Uses all reference cells and installed classifier defaults. It uses a full query unless the estimated query-dependent working set can be reduced by output-equivalent batching. The default 200 discriminating genes is bounded only when fewer than 200 genes overlap. |
| SingleR | Uses all reference cells with the official single-cell reference recommendation `de.method="wilcox"`, `aggr.ref=FALSE`, and fine-tuning and pruning at installed defaults. BiocParallel and the native thread parameter receive the common worker allowance. If needed, classification is batched and pruning is applied once after merging the complete result. |
| Seurat | Uses all reference and query cells with the official `NormalizeData`/`FindVariableFeatures`/`ScaleData` and label-transfer workflow; prediction is not chunked. |
| scANVI | Uses the official scvi-tools reference-mapping recipe with `layers/counts`, 2,000 reference-only HVGs, one SCVI→SCANVI reference checkpoint per seed, and an independent 100-epoch unlabeled query update from that immutable checkpoint. |
| scGraphformer | Loads the official model class and uses the authors' cross-platform settings: a 0.8 reference training fraction, batch size 256, 3,500 reference-selected HVGs, and graph/KNN enabled. The common benchmark adaptation trains once per seed, fixes the reference gene order, constructs reference and query graphs independently, and reuses the model for every query. |
| scGPT | Uses all reference cells, sparse-equivalent binning, mini-batch training, and batch-aligned prediction chunks (2,048 cells with the default batch size). Every query finishes row-ordered binning before tokenization begins; temporary sparse CSR chunks are deleted after that prediction call and are not a reusable cache. |

The 23 GiB value used by the R wrappers and CellTypist prediction is a
best-effort planning threshold,
not a whole-process guarantee. The common pipeline retains one lean reference
object containing expression and required metadata (plus layers explicitly
requested by Seurat and scANVI),
without a second deep training copy. Selective H5AD reads never materialize
unused `raw`, unrelated layers, embeddings, graphs or unstructured metadata.
Per-run AnnData objects share the expression matrices but receive independent
metadata frames. Shared utilities perform only selective loading, input
validation and inference-time metadata isolation;
method-owned transformations, including CellTypist reference normalization,
live in each adapter and run independently inside the timing boundary.
In addition, CellTypist's centered default densifies
the integrated 236,975-by-19,836 reference to about 35 GiB before scaler working
copies. Consequently the official CellTypist Broad run, and potentially other
full-reference runs, require more than 23 GiB. No optimizer or reference
sampling fallback is silently substituted when that happens.

When their memory fallback is needed, SingleR, CHETAH and CellTypist retain
every query cell and concatenate outputs in original order. SingleR performs
global pruning only after all batches are merged. Seurat prediction is not
chunked because its transfer anchors depend on the complete query context.

scGPT uses the tutorial's core settings: a 10% validation split, 3,001 tokens,
batch size 32, all 10 epochs, 51 expression bins, learning rate `1e-4`, dropout
`0.2`, scheduler ratio `0.9`, and an unshuffled training DataLoader. There is
no early stopping; after every configured epoch completes, inference restores
the checkpoint with the lowest validation loss. The common twelve-worker cap,
sparse-equivalent binning and CLS-only execution shortcut are benchmark
implementation/resource settings and must be reported as such.

## Output

A single-query run is saved to `SAVE_PATH/METHOD/RUN_ID/`. A standard query
directory keeps the existing benchmark layout and saves each file to
`SAVE_PATH/DATASET/METHOD/RUN_ID/`:

- `METHOD_metrics.csv`
- `per_class_metrics.csv`
- `predictions.csv`
- `confusion_matrix.csv` and `METHOD_confusion_matrix.png`
- `umap_coordinates.csv`
- prediction, ground-truth, batch and comparison UMAP PNGs when their keys are
  available

Primary accuracy, balanced accuracy, macro/micro/weighted precision, recall
and F1, one-vs-rest AUC, MCC, Cohen's kappa, per-class metrics and confusion
matrices all use the dataset-specific source ontology. When source ground truth
is coarser than the fixed training ontology, accepted child predictions are
projected to the source parent before evaluation; parent AUC scores sum the
accepted child-class probabilities. `compatible_accuracy` remains only as a
backward-compatible alias of primary `accuracy`. Normalized exact-label
equality is reported separately only as `exact_match_accuracy`,
`n_exact_match_correct`, and the per-cell `exact_match_correct` prediction
column. Evaluation coverage and
missing/unsupported and exact/coarse truth counts are also recorded. The
separate CAF task uses its seven exact targets, so it receives no projection or
compatible-label relaxation and its primary and exact-match metrics coincide.
Each method directory also contains `metrics_summary.csv` and
`metrics_aggregate.csv` across repeated runs.

## Immune / non-immune evaluation

Each evaluated non-CAF run also writes `immune/` and `non-immune/`, containing
`METHOD_metrics.csv`, `per_class_metrics.csv`, `confusion_matrix.csv` and
`METHOD_confusion_matrix.png`. Both groups are emitted even when one is empty
(zero cells, NaN quality metrics, header-only matrices). Each method directory
adds `immune_metrics_summary.csv`, `immune_metrics_aggregate.csv`,
`non_immune_metrics_summary.csv` and `non_immune_metrics_aggregate.csv`.
Each group uses the original summary (one row/run) and aggregate (metric
mean/sample SD) formats. No method-level group directories are created;
original summary and aggregate files are unchanged.

Edit the explicit `IMMUNE_LABELS` and `NON_IMMUNE_LABELS` at the top of
[`utils/compartment_evaluation.py`](utils/compartment_evaluation.py).
Cells are selected by **ground
truth**; cell-type labels remain intact, and cross-compartment wrong predictions
remain errors. The existing source-compatible evaluator is reused. Metrics are
conditional on that selected compartment, not binary lineage classification or
full-dataset one-vs-rest averages. Aggregate specificity averages true classes
with positive support. Unsupported/unassigned truths are excluded and reported;
they are never guessed to be non-immune. CAF is skipped. Runtime/RAM/VRAM and
original whole-dataset metrics remain unchanged.

New runs save complete class scores in `annotation_scores.npz`, enabling exact
later AUC recalculation. Old maximum `prediction_score` and global per-class AUC
cannot reconstruct conditional compartment AUC: without a score cache it stays
NaN (`unavailable_no_class_scores`). An original `hard_predictions` AUC can be
reconstructed; `--allow_hard_auc` explicitly requests this different one-hot
score fallback for other historical runs. Single-true-class AUC is undefined.
Completed-run resume supplements derived files without retraining.

From this project directory, using an existing benchmark Python environment:

```bash
python backfill_compartment_metrics.py
python bar_chart_immune.py --results_dir result --sclightgat_results_dir \
    /Group16T/common/ccuc/scLightGAT/scLightGAT/sclightgat_exp_results
python bar_chart_non_immune.py --results_dir result --sclightgat_results_dir \
    /Group16T/common/ccuc/scLightGAT/scLightGAT/sclightgat_exp_results
```

Backfill supports multiple `--results_dir` roots, `--datasets`, `--methods`,
`--no_plots` (CSVs only) and separate `--output_dir`; originals and completion
markers are untouched, and in-progress/incomplete runs are skipped. See `--help`.

The two bar chart scripts are standalone and live only alongside the original
`bar_chart.py`, with no plotting helper in utils. Edit `METRIC_GROUPS` at the
top of each script to select metrics. Outputs are PNGs only. Dataset error bars
are sample SD across runs; `Overall` equally weights available non-CAF dataset
means and has no error bars, matching the original chart. Empty/missing metrics
are never zero-filled. After changing definitions, rerun backfill.
