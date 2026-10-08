## Data information

### scCobra_immune
```
data = "/Group16T/raw_data/scCobra/Immune_ALL_human.h5ad"
out_path = "/Group16T/common/ccuc/scCobra/result/immune/"
batch = 'batch'
celltype = 'final_annotation'
```

### scCobra_lung
```
data = "/Group16T/raw_data/scCobra/Lung_atlas_public.h5ad"
out_path = "/Group16T/common/ccuc/scCobra/result/lung/"
batch = 'batch'
celltype = 'cell_type'
```

### scCobra_pancreas
```
data = "/Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad"
out_path = "/Group16T/common/ccuc/scCobra/result/pancreas/"
batch = 'tech'
celltype = 'celltype'
```

## Preprocessing policy

The shared pipeline does not normalize, filter, select features, or compute a
tool input representation by default. Each method wrapper owns its preprocessing
and follows either the installed tool defaults or the method's official workflow:

- **Raw/PCA:** the fixed benchmark design (QC, log-normalization, 2,000 HVGs,
  PCA); this is intentionally not a tool default.
- **Harmony:** library-size normalization, log1p, the union of 2,000 variable
  features selected per batch with the Seurat-vst equivalent, scaling, and 20
  PCs before the `harmonypy` implementation, following the official Harmony
  Seurat vignette.
- **Scanorama:** total-count normalization and log1p on all genes; Scanorama's
  `hvg=None` default is retained.
- **scVI/scANVI:** batch-aware 2,000 HVGs selected from the source-expression
  layer with `seurat_v3`; the models read `layers["counts"]`. Direct scANVI
  initialization is an intentional experimental design choice.
- **Seurat:** source expression is passed to Seurat, which performs its default
  LogNormalize/vst workflow and CCA integration with 30 dimensions.
- **SMNN:** full-gene source expression and SMNN 1.2.1 correction defaults. The
  multi-batch implementation is patched and must be reported as such.
- **scDML:** native `scDMLModel.preprocess()` and `integrate()` defaults.  The
  installed 0.0.1 preprocessor does not implement the extra low-count filters
  described in the paper, so the benchmark deliberately follows the package
  API rather than adding non-API QC.
- **scCobra:** native RNA preprocessing defaults, including its own QC and
  MaxAbs scaling.
- **scCM:** paper workflow (300--6,000 detected features, at most 15% MT,
  normalization/log1p, and 3,000 HVGs); training follows the defaults reported
  in the paper Methods (batch size 512, learning rate 1e-6, momentum 0.99,
  temperature 0.8, five warm-up epochs, and ten total epochs).  The vendored
  launcher includes the minimal positional-argument fix needed to pass both
  momentum and temperature to the model as documented upstream.

An existing `layers["counts"]` is treated as the experiment-designated raw-count
source matrix, including fractional values in the pancreas input and the
non-integer Villani batch in the immune input. Villani must be retained; no
wrapper may remove it based on an integer-count check. Because scCobra and scCM
apply their own documented QC, their evaluated cell sets may differ from other
methods.

## Method references and configuration rationale

The links below separate three kinds of evidence: the method paper explains the
algorithm, official documentation or source code establishes API defaults, and
the benchmark note records any deliberate local choice. A local experimental
choice or patch is not presented as an upstream default.

### Harmony

Input preparation follows the [official Harmony Seurat
vignette](https://github.com/immunogenomics/harmony/blob/master/doc/Seurat.Rmd):
library-size normalization and log transformation, 2,000 variable features
selected within each batch and combined as a union, feature scaling, and a
20-component PCA. The Python wrapper implements the Seurat-v3-compatible VST
step with Scanpy and runs
[`harmonypy`](https://github.com/slowkow/harmonypy) on the PCA embedding. These
steps construct Harmony's input; they are not preprocessing performed inside
the Harmony algorithm. Method reference: [Korsunsky et al., *Nature Methods*
(2019)](https://doi.org/10.1038/s41592-019-0619-0).

### Scanorama

Source expression is normalized to 10,000 counts per cell and log1p-transformed,
following the [published Scanorama
protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC11361826/). The full gene set
is passed to `integrate_scanpy()` and the `hvg` argument is omitted. The
[official source](https://github.com/brianhie/scanorama/blob/master/scanorama/scanorama.py)
defines the forwarded `integrate()` default as `hvg=None`, so the benchmark does
not request Scanorama-specific HVG filtering. Method reference: [Hie et al.,
*Nature Biotechnology* (2019)](https://doi.org/10.1038/s41587-019-0113-3);
[official repository](https://github.com/brianhie/scanorama).

### Seurat

Each batch uses [`NormalizeData()`](https://satijalab.org/seurat/reference/normalizedata)
(`LogNormalize`, scale factor 10,000) and
[`FindVariableFeatures()`](https://satijalab.org/seurat/reference/findvariablefeatures)
(`vst`, 2,000 features), followed by 2,000
[integration features](https://satijalab.org/seurat/reference/selectintegrationfeatures).
The benchmark uses the anchor workflow: the documented defaults of
[`FindIntegrationAnchors()`](https://satijalab.org/seurat/reference/findintegrationanchors)
are CCA and dimensions 1:30, and
[`IntegrateData()`](https://satijalab.org/seurat/reference/integratedata) also
uses dimensions 1:30. The integrated assay is scaled and represented by 30
PCs. Method references: [Stuart et al., *Cell*
(2019)](https://doi.org/10.1016/j.cell.2019.05.031) and [Butler et al., *Nature
Biotechnology* (2018)](https://doi.org/10.1038/nbt.4096).

### scANVI

The shared scVI/scANVI preparation selects 2,000 batch-aware HVGs from the
source-expression layer with `seurat_v3`, while the model is explicitly
registered against `layers["counts"]`. This benchmark intentionally constructs
`SCANVI(adata)` directly and does not pretrain an scVI model. Direct
construction is supported by the [official `SCANVI`
API](https://docs.scvi-tools.org/en/latest/api/reference/scvi.model.SCANVI.html),
but official tutorials commonly pretrain scVI and initialize scANVI with
`from_scvi_model()`; see the [official seed-labeling
tutorial](https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/seed_labeling.html).
Direct initialization is therefore a deliberate experimental design, not the
tutorial initialization workflow. Method reference: [Xu et al., *Molecular
Systems Biology* (2021)](https://doi.org/10.15252/msb.20209620).

### scDML

The wrapper calls native `scDMLModel.preprocess()` with the installed package
defaults: total-count normalization to 10,000, log1p, 1,000 HVGs, package-side
per-batch feature scaling, 100 PCs, and Louvain resolution 3. Integration also
retains the [official source
defaults](https://github.com/eleozzr/scDML/blob/main/scDML/main.py)
`ncluster_list=[3]` and `expect_num_cluster=None`. The former is a merge
candidate list; it does **not** fix the expected number of clusters to three.
With `expect_num_cluster=None`, scDML uses its eigenvalue-gap estimate. Method
reference: [Yu et al., *Nature Communications*
(2023)](https://doi.org/10.1038/s41467-023-36635-5); [official
repository](https://github.com/eleozzr/scDML).

### scCobra

The benchmark follows the official package's [native RNA preprocessing
source](https://github.com/mcgilldinglab/scCobra/blob/main/scCobra/data.py#L186-L269):
batches with at most 10 cells are removed; `ERCC`, `MT-`, and `mt-` genes are
excluded; cells with fewer than 600 detected genes and genes found in fewer
than 3 cells are filtered; counts are normalized to 10,000 and log1p-transformed;
2,000 batch-aware HVGs are selected; and the matrix is MaxAbs-scaled. These
deterministic steps run before timing, after which the wrapper calls
`scCobra(..., processed=True)`. The [processed-input
branch](https://github.com/mcgilldinglab/scCobra/blob/main/scCobra/data.py#L619-L640)
skips the package's filtering, normalization, and HVG selection and reapplies
only MaxAbs scaling; this does not change the already nonnegative,
columnwise-MaxAbs-scaled input.

The [published scCobra Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11825689/)
describe different analysis QC thresholds. This benchmark deliberately treats
the package's `preprocessing_rna()` implementation as the operational default
rather than mixing the paper's analysis pipeline into the package API.
Method reference: [Zhao et al., *Communications Biology*
(2025)](https://doi.org/10.1038/s42003-025-07692-x); [official
repository](https://github.com/mcgilldinglab/scCobra).

### scCM

The wrapper follows the [scCM paper](https://doi.org/10.1038/s42003-024-06813-2):
it retains cells with 300--6,000 detected features and at most 15% mitochondrial
expression, normalizes each cell to 10,000 counts, applies log1p, and selects
3,000 global HVGs. The published sentence describing the mitochondrial filter
appears to reverse the exclusion inequality; the benchmark implements the
standard intended interpretation, retaining cells at or below 15% and excluding
cells above it.

Training uses the paper's stated defaults: a 128-dimensional embedding,
temperature 0.8, momentum 0.99, learning rate 1e-6, batch size 512 (capped by
the number of available cells), five warm-up epochs, and ten total epochs. The
vendored launcher fixes the upstream positional-argument mismatch so both
momentum and temperature reach the model as documented. See the [official
launcher/source](https://github.com/farry92/ScCM/blob/main/scCM.py), [official
repository](https://github.com/farry92/ScCM), and [archived
release](https://doi.org/10.5281/zenodo.13119941). Input H5AD files and scCM
output/checkpoint directories are intentionally retained after a run for
debugging; retention is a benchmark operational policy, not an upstream model
default.

### SMNN

SMNN is supervised: benchmark cell-type annotations constrain mutual-nearest-
neighbor searches to matching cell types. The correction retains the [official
SMNN 1.2.1 nominal defaults](https://rdrr.io/github/yycunc/SMNN/man/SMNNcorrect.html):
`k=20`, `sigma=1`, `cos.norm.in=TRUE`, `cos.norm.out=TRUE`, `var.adj=TRUE`,
`correct.others=FALSE`, and full-gene correction (`subset.genes=NULL`).

The local code is **not** an unmodified `SMNNcorrect()` call and must be reported
as a patched SMNN 1.2.1 multi-batch implementation. It corrects batches
sequentially in input order, updates labels with the accumulated reference,
caps `k` for cell types with fewer than 20 available cells, and replaces the
package's serial variance-adjustment loop with the parallel mnnpy helper while
retaining the package correction-vector function. Correction uses all common
genes; the 2,000 unsupervised integration features and final 50-PC embedding are
downstream benchmark representation steps, not SMNN defaults. Method reference:
[Yang et al., *Briefings in Bioinformatics*
(2021)](https://doi.org/10.1093/bib/bbaa097); [official
repository/source](https://github.com/yycunc/SMNN).

## Method install
### Basic
```
pip install scanpy
pip install scipy
pip install igraph leidenalg
conda install pyarrow fastparquet
```

### Harmony
```
pip install harmonypy
```

### scVi
```
pip install scvi-tools
```

### Scanorama
```
pip install scanorama
```

### SMNN
```
pip install mnnpy
```

The SMNN wrapper also requires the R packages `SMNN`, `Seurat`,
`SeuratObject`, `Matrix`, `argparse`, and `reticulate` in its execution
environment.

SMNN is a supervised method. The benchmark passes `--celltype_key` to SMNN
for within-cell-type mutual-nearest-neighbor matching, as specified by the
published method. The same annotation is also used by the common evaluation
pipeline, so SMNN scores must be reported as supervised results and should not
be interpreted as directly equivalent to the unsupervised tools.

The wrapper keeps the official defaults (`k=20`, `sigma=1`, cosine
normalization, and variance adjustment), corrects the full-gene source matrix,
and uses 2,000 unsupervised integration features only for the final
50-dimensional PCA embedding. The multi-batch implementation contains local
execution fixes and must be described as a patched SMNN 1.2.1 implementation.

### Seurat
#### Conda
```
conda install -c conda-forge r-base==4.3 r-essential
conda install -c conda-forge \
  r-matrix r-mass r-seurat r-seuratobject r-sctransform r-fitdistrplus r-igraph r-irlba r-rspectra r-uwot
```
#### R
```
install.packages(c("spatstat.geom", "spatstat.explore", "argparse", "reticulate", "aricode", "feather"))
```

### scDML
```
conda create -n scDML python==3.8.12
git clone https://github.com/eleozzr/scDML
cd scDML
pip install .
pip install igraph leidenalg 
```

### scCobra
```
conda create -n scCobra conda-forge::python=3.10 conda-forge::ipykernel 
pip install scanpy scib
pip3 install torch torchvision torchaudio torch_optimizer
pip3 install igraph leidenalg
git clone https://github.com/mcgilldinglab/scCobra.git
```
