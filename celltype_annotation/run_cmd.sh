#!/usr/bin/env bash
set -u -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
overall_status=0

bash "$script_dir/run_script.sh" \
    --train_path /Group16T/common/ccuc/scLightGAT/preprocessing/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/preprocessing/scLightGAT_data/Independent_testing/ \
    --save_path "/Group16T/common/ccuc/scLightGAT/scLightGAT/result" \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch None \
    --test_batch_map "GSE115978=samples,GSE123139=Sample.Short.Name,GSE153935=orig.ident,GSE166555=sample_id,Zhengsorted=None,lung_full=None,sapiens_full=donor" \
    --n_jobs 12 \
    --n_runs 5 || overall_status=1

bash "$script_dir/run_script.sh" \
    --train_path /Group16T/common/ccuc/scLightGAT/preprocessing/scLightGAT_data/caf.data/caf_train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/preprocessing/scLightGAT_data/caf.data/caf_test.h5ad \
    --save_path "/Group16T/common/ccuc/scLightGAT/scLightGAT/result/CAF" \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch SampleID \
    --test_batch None \
    --n_jobs 12 \
    --n_runs 5 || overall_status=1

subtype_data=/Group16T/common/ccuc/scLightGAT/preprocessing/scLightGAT_data/Subtype
subtype_result="/Group16T/common/ccuc/scLightGAT/scLightGAT/result/Subtype"

for broad in B_cell CD4_T_cell CD8_T_cell DC Plasma_cell; do
    bash "$script_dir/run_script.sh" \
        --train_path "$subtype_data/${broad}_train.h5ad" \
        --test_path "$subtype_data/${broad}_test.h5ad" \
        --save_path "$subtype_result/$broad/" \
        --train_celltype Celltype_subtraining \
        --test_celltype Celltype_subtraining \
        --train_batch Data_source \
        --test_batch Data_source \
        --n_jobs 12 \
        --n_runs 5 || overall_status=1
done

if ((overall_status != 0)); then
    echo "One or more benchmark groups failed; inspect the logs above." >&2
    exit 1
fi
