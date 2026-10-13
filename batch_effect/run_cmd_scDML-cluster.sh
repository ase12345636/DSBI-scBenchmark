#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash run_script_scDML-cluster.sh \
    --dataset_path /Group16T/raw_data/scCobra/Immune_ALL_human.h5ad \
    --save_path /Group16T/common/ccuc/scSCI/result/immune \
    --batch_key batch \
    --celltype_key final_annotation \
    --n_runs 5

bash run_script_scDML-cluster.sh \
    --dataset_path /Group16T/raw_data/scCobra/Lung_atlas_public.h5ad \
    --save_path /Group16T/common/ccuc/scSCI/result/lung \
    --batch_key batch \
    --celltype_key cell_type \
    --n_runs 5

bash run_script_scDML-cluster.sh \
    --dataset_path /Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad \
    --save_path /Group16T/common/ccuc/scSCI/result/pancreas \
    --batch_key tech \
    --celltype_key celltype \
    --n_runs 5
