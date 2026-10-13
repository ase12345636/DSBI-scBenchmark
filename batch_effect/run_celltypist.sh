#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/experiments/celltypist_annotaion/data/celltypist_immune.h5ad \
    --save_path /Group16T/common/ccuc/scQuartz/result/immune_celltypist \
    --batch_key batch \
    --celltype_key celltypist_label \
    --n_runs 5
