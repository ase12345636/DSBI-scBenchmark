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
pip install harmony-pytorch
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
