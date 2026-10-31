# Single-Cell Batch Correction Benchmark

This repository provides a unified framework for benchmarking a representative collection of single-cell batch-effect correction and data-integration methods. It includes an uncorrected PCA baseline, conventional embedding- and
neighborhood-based methods, deep-learning models, self-supervised approaches, and hybrid integration pipelines.

Each method uses a common command-line interface and produces a low-dimensional cell representation for a shared downstream evaluation workflow. The benchmark supports repeated runs and evaluates biological conservation, batch mixing, runtime, and peak RAM/VRAM usage.

The framework standardizes data loading, shared quality control (QC), execution, output organization, and downstream evaluation while allowing each method to retain its own documented preprocessing, feature-selection, and training workflow based on the corresponding publication, official implementation, tool defaults, or an explicitly documented benchmark choice. Shared QC and wrapper-level preprocessing are performed before timing, whereas preprocessing executed internally by a method remains part of its timed call. Downstream evaluation is excluded from runtime and memory measurements.

The shared utilities provide reusable interfaces for data handling, repeated execution, evaluation, and result reporting. New methods can be added by implementing the wrapper interface defined in [`utils/pipeline.py`](utils/pipeline.py), without duplicating the common benchmark logic.

This benchmark is intended to provide a reproducible comparison framework rather than an exhaustive catalog of available integration methods.

## Included Methods

Methods are grouped by their primary integration mechanism. Because several methods combine multiple model architectures or training objectives, these categories describe their dominant design and are not necessarily mutually exclusive.

In the table below, cell-type supervision refers to the use of cell-type annotations or ground-truth-derived information during integration. The use of batch identifiers alone is not considered cell-type supervision.

<table>
<thead>
  <tr>
    <th>Major Class</th>
    <th>Minor Class</th>
    <th>Name</th>
    <th>Input Preprocessing</th>
    <th>Cell-Type Supervision</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td>Raw<br>
        (Uncorrected)
    </td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
      PCA (50)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td>Low-Dimensional</td>
    <td></td>
    <td><a href="https://doi.org/10.1038/s41592-019-0619-0">Harmony</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
      Scale (Max 10)<br>
      PCA (20)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="3">Neighbor-Based Integration</td>
    <td rowspan="2">MNN Alignment</td>
    <td><a href="https://doi.org/10.1038/s41587-019-0113-3">Scanorama</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1093/bib/bbaa097">SMNN</a></td>
    <td>
      Cosine Normalization<br>
    </td>
    <td><b>Full Cell-Type</b></td>
  </tr>
  <tr>
    <td>Anchor Alignment</td>
    <td><a href="https://doi.org/10.1016/j.cell.2019.05.031">Seurat CCA</a></td>
    <td>
      LogNormalize (10,000)<br>
      HVG (2000)<br>
      Integration features (2000)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="14">Deep Learning Integration</td>
    <td rowspan="3">Autoencoder-Based Models</td>
    <td><a href="https://doi.org/10.1038/s41592-018-0229-2">scVI</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.15252/msb.20209620">scANVI</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
    </td>
    <td><b>Full Cell-Type</b></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1038/s41467-022-33758-z">SCALEX</a></td>
    <td>
      Remove MT/ERCC genes<br>
      Filter genes (min. 3 cells)<br>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
      Scale (MaxAbs)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td>Adversarial Alignment</td>
    <td><a href="https://doi.org/10.1038/s41467-023-43590-8">scDREAMER</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">Deep Metric Learning</td>
    <td><a href="https://doi.org/10.1038/s41467-023-36635-5">scDML</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (1000)<br>
      Scale<br>
      PCA (100)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1038/s41467-023-36635-5">scDML-cluster</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (1000)<br>
      Scale<br>
      PCA (100)<br>
    </td>
    <td><b>Oracle-K</b></td>
  </tr>
  <tr>
    <td>Feature Decorrelation</td>
    <td><a href="https://doi.org/10.1038/s41598-026-50586-z">scDecorr</a></td>
    <td>
      Filter genes (min. 3 cells)<br>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">Contrastive Learning</td>
    <td><a href="https://doi.org/10.1038/s42003-025-07692-x">scCobra</a></td>
    <td>
      Remove batches (&le;10 cells)<br>
      Normalize (10,000)<br>
      log1p<br>
      HVG (2000)<br>
      Scale (MaxAbs)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1093/bib/bbac377">CLEAR</a></td>
    <td>
      Normalize (10,000)<br>
      Scale (Max 10)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1038/s41587-025-02950-z">CONCORD</a></td>
    <td>
      HVG (2000)<br>
      Normalize (median library size)<br>
      log1p<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1038/s42003-024-06813-2">scCM</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (3000)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="3">Disentanglement</td>
    <td><a href="https://doi.org/10.1101/2025.03.15.643447">BioBatchNet</a></td>
    <td>
      HVG (2000)<br>
      Normalize (10,000)<br>
      log1p<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1101/2025.11.03.683998">FADVI</a></td>
    <td>
      HVG (2000)<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://doi.org/10.1093/nar/gkab978">SCIDRL</a></td>
    <td>
      HVG (1000 per batch)<br>
      Normalize (10,000)<br>
      log1p<br>
      Min-max Scale<br>
    </td>
    <td></td>
  </tr>
  <tr>
    <td>Additional Processing</td>
    <td>Vector Quantization</td>
    <td><a href="https://doi.org/10.1038/s41467-025-56424-6">MetaQ</a></td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (Adaptive)<br>
      Scale (Max 10)<br>
      PCA (50)<br>
    </td>
    <td></td>
  </tr>
</tbody>
</table>

## Evaluation Protocol

All methods return a low-dimensional cell representation that is evaluated using the shared downstream workflow. Cells are aligned by `adata.obs_names`. If method-specific preprocessing removes cells, metrics are computed using the intersection of the shared post-QC reference and the method output.

Cell-type annotations are required for evaluation. They are used for clustering-resolution selection and biological-conservation metrics. During integration or model fitting, cell-type labels or ground-truth-derived information are used only by methods explicitly identified in the table above.

For clustering-based evaluation, a neighbor graph is constructed from each method's integrated representation. Leiden resolutions from 0.1 to 2.0 in increments of 0.1 are evaluated, and the resolution with the highest NMI against the reference cell-type labels is selected. The reported NMI and ARI, together with the Leiden visualization, use this selected clustering.

## Prerequisites

- Linux with Bash
- Git
- A Conda distribution, such as Miniconda or Anaconda; each environment installs its required Python version.

### Input requirements

Input data must be provided as an AnnData (`.h5ad`) file. The input object must contain:

- an expression matrix in `adata.X`;
- batch identifiers in `adata.obs`, specified using `--batch_key`;
- cell-type annotations in `adata.obs`, specified using `--celltype_key`;
- gene identifiers in `adata.var_names`.

Shared QC uses `adata.layers["counts"]` when available and otherwise uses `adata.X`. The matrix used for shared QC must contain nonnegative, untransformed source expression values and should retain the full gene set. Log-normalized, scaled, batch-corrected, or HVG-only matrices should not be supplied as the sole source matrix. Some count-based wrappers impose additional input validation after shared QC.

Batch and cell-type columns should not contain missing values. Cell identifiers should also remain stable so that cells can be aligned correctly before and after integration.

### Quality Control

A shared QC step is applied before method-specific preprocessing. The current default filters:

- retain cells with at least 200 detected genes;
- retain genes detected in at least three cells;
- retain cells with less than 5% mitochondrial counts.

Mitochondrial genes are identified from gene names beginning with `MT-`, case-insensitively. The default rule therefore expects gene symbols with this prefix. Datasets using Ensembl-only or other identifiers should map them to gene symbols or modify the mitochondrial annotation logic in [`utils/dataset.py`](utils/dataset.py); otherwise, mitochondrial genes will not be detected by the default rule.

QC thresholds are dataset dependent and should be reviewed before applying the benchmark to a new dataset. The shared implementation and default values are defined in [`utils/dataset.py`](utils/dataset.py). Adjust the parameters or filtering logic there when a dataset requires different criteria. Individual wrappers may apply additional method-specific QC or filtering after the shared QC step.

## Environment Installation

This benchmark uses multiple Conda environments to isolate incompatible method-specific dependencies. Create all environments before running `run_script.sh`, which executes every included method in sequence. When running an individual wrapper directly, only that method's environment is required. The launcher activates the appropriate environment for each method; it does not create or install environments automatically.

The shared `sctools` environment supports the largest group of methods with compatible dependencies. Methods that require specific Python, R, TensorFlow, PyTorch, or CUDA versions use separate environments. Seurat and SMNN execute their R implementations through Python wrappers so that they can use the same command-line interface, output structure, and evaluation pipeline as the Python-based methods. Each environment also includes the common packages required by the benchmark's data-loading, evaluation, and resource-monitoring utilities, even when those packages are not direct dependencies of the original method.

Use the commands below to create the required environments with the exact names expected by `run_script.sh`.

An NVIDIA GPU with a compatible driver is required for scCM and CLEAR.

Before installing or running the benchmark, set the following variable in every shell session to prevent user-site packages from leaking into the Conda environments:

```bash
export PYTHONNOUSERSITE=1
```

### sctools: Raw, Harmony, scVI, Scanorama, scANVI, scCM, CONCORD, BioBatchNet, and FADVI

```bash
conda create -n sctools --override-channels -c conda-forge \
  python=3.10 pip -y
conda activate sctools

python -m pip install \
  torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  scanpy==1.11.5 anndata==0.11.4 numpy==1.26.4 pandas==2.3.3 \
  scipy==1.14.1 scikit-learn==1.7.2 matplotlib==3.10.9 \
  igraph==1.0.0 leidenalg==0.12.0 h5py==3.16.0 networkx==3.4.2 \
  harmonypy==2.0.0 scanorama==1.7.4 scvi-tools==1.3.3 \
  scikit-misc==0.5.2 scib==1.1.7 \
  pytorch-lightning==2.6.5 lightning==2.6.5 \
  PyYAML==6.0.3 tensorboard==2.21.0 psutil==7.2.2 \
  pillow==12.3.0 seaborn==0.13.2 tqdm==4.69.0 \
  umap-learn==0.5.12 numba==0.66.0 llvmlite==0.48.0

python -m pip check
```

### Seurat

```bash
conda create -n Seurat --override-channels -c conda-forge \
  python=3.10 pip \
  r-base=4.3.3 r-seurat=5.3.0 r-seuratobject=5.2.0 \
  r-matrix=1.6_5 r-argparse=2.2.5 r-future=1.67.0 -y
conda activate Seurat

python -m pip install \
  scanpy==1.11.5 anndata==0.11.4 numpy==1.26.4 pandas==2.3.3 \
  scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.9 \
  igraph==1.0.0 leidenalg==0.12.0 h5py==3.16.0 networkx==3.4.2

python -m pip check
Rscript -e 'library(Seurat); library(argparse); library(Matrix); library(future); sessionInfo()'
```

### SCIDRL

```bash
conda create -n SCIDRL --override-channels -c conda-forge \
  python=3.9 pip=24.0 cudatoolkit=11.8.0 \
  numpy=1.23.5 pandas=1.5.3 scipy=1.10.1 scikit-learn=1.2.2 \
  anndata=0.8.0 h5py=3.8.0 scanpy=1.9.3 matplotlib=3.7.1 \
  seaborn=0.12.2 tqdm=4.66.2 python-igraph=0.10.4 \
  leidenalg=0.9.1 umap-learn=0.5.3 numba=0.56.4 llvmlite=0.39.1 \
  pynndescent=0.5.10 networkx=3.2.1 -y
conda activate SCIDRL

python -m pip install \
  tensorflow==2.13.1 keras==2.13.1 scikit-misc==0.3.1 \
  nvidia-cudnn-cu11==8.6.0.163 nvidia-cublas-cu11==11.11.3.6

conda env config vars set \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib"
conda deactivate
conda activate SCIDRL

python -m pip check
```

### scCobra

```bash
conda create -n scCobra --override-channels -c conda-forge \
  python=3.10 pip -y
conda activate scCobra

python -m pip install torch==2.9.1 torch-optimizer==0.3.0
python -m pip install \
  scanpy==1.11.5 anndata==0.11.4 numpy==1.26.4 pandas==2.3.3 \
  scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8 \
  igraph==1.0.0 leidenalg==0.11.0 h5py==3.15.1 networkx==3.4.2 \
  seaborn==0.13.2 tqdm==4.67.1

python -m pip check
```

### scDML

```bash
conda create -n scDML --override-channels -c conda-forge \
  python=3.8.12 pip=24.2 git cxx-compiler -y
conda activate scDML

python -m pip install \
  numpy==1.18.1 pandas==1.3.5 scipy==1.8.1 scikit-learn==1.3.2 \
  scanpy==1.7.2 anndata==0.7.6 matplotlib==3.5.3 h5py==3.11.0 \
  networkx==2.5 python-igraph==0.8.2 leidenalg==0.8.2 \
  numba==0.51.2 llvmlite==0.34.0 numexpr==2.7.1 \
  numpy-groupies==0.9.14 tables==3.6.1 umap-learn==0.4.6 \
  louvain==0.7.0 plotly==5.2.2 hnswlib==0.5.2 annoy==1.17.0 \
  ipykernel==5.5.6 nbformat==5.1.3 seaborn==0.12.2 tqdm==4.67.1 \
  joblib==1.4.2 threadpoolctl==3.5.0 pillow==10.4.0

python -m pip install \
  torch==1.10.1+cu111 torchvision==0.11.2+cu111 \
  --extra-index-url https://download.pytorch.org/whl/cu111
python -m pip install pytorch-metric-learning==0.9.95
python -m pip install --no-deps \
  'git+https://github.com/eleozzr/scDML.git@d762dd75557c0dbb2d3039174d69a18195842b93'

python -m pip check
```

### SCALEX

```bash
conda create -n SCALEX --override-channels -c conda-forge \
  python=3.10 pip numpy=1.26.4 pandas=2.2.2 scipy=1.13.0 \
  scikit-learn=1.4.2 anndata=0.10.8 h5py=3.11.0 scanpy=1.10.1 \
  matplotlib=3.8.4 seaborn=0.13.2 tqdm=4.66.2 \
  python-igraph=0.11.8 leidenalg=0.10.2 umap-learn=0.5.7 \
  networkx=3.4.2 -y
conda activate SCALEX

python -m pip install \
  torch==2.2.2+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip check
```

### scDREAMER

```bash
conda create -n scDREAMER --override-channels -c conda-forge \
  python=3.9 pip cudatoolkit=11.8.0 \
  numpy=1.23.5 pandas=1.5.3 scipy=1.10.1 scikit-learn=1.2.2 \
  anndata=0.8.0 h5py=3.8.0 scanpy=1.9.3 matplotlib=3.7.1 \
  seaborn=0.12.2 tqdm=4.65.0 python-igraph=0.10.4 \
  leidenalg=0.9.1 umap-learn=0.5.3 numba=0.56.4 llvmlite=0.39.1 \
  pynndescent=0.5.8 networkx=3.2.1 -y
conda activate scDREAMER

python -m pip install \
  tensorflow==2.12.0 keras==2.12.0 \
  nvidia-cudnn-cu11==8.6.0.163 nvidia-cublas-cu11==11.11.3.6

conda env config vars set \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib"
conda deactivate
conda activate scDREAMER

python -m pip check
```

### CLEAR

```bash
conda create -n CLEAR --override-channels -c conda-forge \
  python=3.9 pip numpy=1.23.5 pandas=1.5.3 scipy=1.10.1 \
  scikit-learn=1.2.2 anndata=0.8.0 h5py=3.8.0 scanpy=1.9.3 \
  matplotlib=3.7.1 seaborn=0.12.2 tqdm=4.66.2 pillow=9.5.0 \
  python-igraph=0.10.4 leidenalg=0.9.1 umap-learn=0.5.3 \
  numba=0.56.4 llvmlite=0.39.1 pynndescent=0.5.10 \
  networkx=3.2.1 -y
conda activate CLEAR

python -m pip install \
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip check
```

### MetaQ

```bash
conda create -n MetaQ --override-channels -c conda-forge \
  python=3.11.6 pip numpy=1.26.0 pandas=2.1.4 scipy=1.11.3 \
  scikit-learn=1.1.3 anndata=0.10.8 h5py=3.10.0 scanpy=1.9.6 \
  zarr=2.18.3 numcodecs=0.12.1 matplotlib=3.8.4 seaborn=0.13.2 \
  tqdm=4.66.2 python-igraph=0.11.8 leidenalg=0.10.2 \
  umap-learn=0.5.7 networkx=3.6.1 -y
conda activate MetaQ

python -m pip install \
  torch==2.1.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  harmonypy==0.0.6 alive-progress==3.1.5 einops==0.7.0
python -m pip check
```

### scDecorr

```bash
conda create -n scDecorr --override-channels -c conda-forge \
  python=3.9 pip numpy=1.23.5 pandas=1.5.3 scipy=1.10.1 \
  scikit-learn=1.2.2 anndata=0.8.0 h5py=3.8.0 scanpy=1.9.3 \
  matplotlib=3.7.1 seaborn=0.12.2 tqdm=4.66.2 \
  python-igraph=0.10.4 leidenalg=0.9.1 umap-learn=0.5.3 \
  numba=0.56.4 llvmlite=0.39.1 pynndescent=0.5.10 \
  pyyaml=6.0.2 prettytable=3.16.0 pillow=11.3.0 \
  networkx=3.2.1 -y
conda activate scDecorr

python -m pip install \
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install scikit-misc==0.3.1
python -m pip check
```

### SMNN

```bash
conda create -n SMNN --override-channels -c conda-forge -c bioconda \
  python=3.10 pip numpy=1.24.4 scipy=1.8.1 numba=0.57.1 \
  llvmlite=0.40.1 h5py=3.8.0 pandas=1.5.3 anndata=0.8.0 \
  mnnpy=0.1.9.5 \
  r-base=4.3.3 r-remotes=2.5.0 r-matrix=1.6_5 r-dplyr=1.1.4 \
  r-reshape=0.8.10 r-reticulate=1.43.0 r-argparse=2.2.5 \
  r-seurat=5.3.0 r-seuratobject=5.2.0 bioconductor-s4vectors=0.40.2 -y
conda activate SMNN

python -m pip install \
  scanpy==1.11.5 scikit-learn==1.7.2 matplotlib==3.10.9 \
  igraph==1.0.0 leidenalg==0.12.0 networkx==3.4.2 \
  numpy==1.24.4 scipy==1.8.1 numba==0.57.1 llvmlite==0.40.1 \
  h5py==3.8.0 pandas==1.5.3 anndata==0.8.0

Rscript -e 'remotes::install_github("yycunc/SMNN@a7d79a9ee80af177033752410f004669ac6f1240", dependencies=FALSE, upgrade="never")'

python -m pip check
Rscript -e 'library(SMNN); library(Seurat); library(reticulate); stopifnot(as.character(packageVersion("SMNN")) == "1.2.1")'
```

## Usage

### Running the Benchmark

Run the benchmark from the repository root after completing the installation steps above.

If Conda is not available in `PATH`, specify its installation directory:

```bash
export CONDA_BASE="<conda_path>"
```

Run the benchmark with the following command. Replace each value enclosed in angle brackets (`<...>`) with the corresponding value for your dataset:

```bash
bash run_script.sh \
  --dataset_path "<dataset_path>" \
  --save_path "<result_path>" \
  --batch_key "<obs_batch_key>" \
  --celltype_key "<obs_celltype_key>" \
  --n_runs "<repeat_time>"
```

`--dataset_path` and `--save_path` are required. `--batch_key`, `--celltype_key`, and `--n_runs` are optional and default to `batch`, `celltype`, and `5`, respectively. The batch and cell-type keys must correspond to columns in `adata.obs`.

The launcher runs all included methods in sequence. Raw is evaluated once, whereas the remaining methods use `--n_runs`. Runs with complete output files are skipped automatically when the command is repeated.

### Adding New Methods

Add each method through a thin wrapper that reuses the shared pipeline. The wrapper should contain only method-specific preprocessing and integration; shared QC, repeated execution, resource monitoring, output organization, and evaluation are handled by [`utils/pipeline.py`](utils/pipeline.py). Use [`Harmony/Harmony.py`](Harmony/Harmony.py) as a Python example and [`Seurat/Seurat.py`](Seurat/Seurat.py) as an external-process example.

A wrapper must:

- create `<Method>/<Method>.py` under the repository root and import `create_argument_parser` and `run_integration_pipeline` from `utils.pipeline`;

- create the common command-line interface with `create_argument_parser("<Method>")`; method-specific arguments may be added before calling `parse_args()`;

- optionally implement deterministic preprocessing as `preprocess_method(adata, celltype_key, batch_key) -> AnnData`; this function runs once before repeated runs and before runtime and memory measurement, so seed-dependent work must remain inside the integration function;

- implement `integrate_method(adata, seed, run_out_path, args) -> AnnData` and use the supplied `seed` for method-specific randomness; and

- store a finite two-dimensional representation in `adata.obsm[embed_key]`, with one row per returned cell. If cells are filtered or reordered, their `adata.obs_names` must remain correctly aligned.

Connect the wrapper to the shared pipeline as follows:

```python
parser = create_argument_parser("<Method>")
args = parser.parse_args()

run_integration_pipeline(
    "<Method>",
    integrate_method,
    "X_<Method>",
    args,
    run_pca=False,
    preprocess_fn=preprocess_method,
)
```

Omit `preprocess_fn` when no method-specific preprocessing is required. Set `run_pca=True` only when the method requires the shared pipeline to create `adata.obsm["X_pca"]` before timing. Use the same method name consistently; it determines the result directory and metrics filename prefix.

Finally, create and document a compatible Conda environment containing the method-specific and common pipeline dependencies. Add its activation and wrapper command to `run_script.sh`, and forward the standard dataset, output, annotation-key, and run-count arguments. Also update the Included Methods and Environment Installation sections. Before adding the method to the full launcher, run the wrapper once with `--run_times 1` and confirm that the metrics, UMAP coordinates, resolution search, and UMAP figures are produced.