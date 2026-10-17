bash run_script.sh \
    --dataset_path /Group16T/raw_data/scCobra/Immune_ALL_human.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/immune \
    --batch_key batch \
    --celltype_key final_annotation \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/raw_data/scCobra/Lung_atlas_public.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/lung \
    --batch_key batch \
    --celltype_key cell_type \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/pancreas \
    --batch_key tech \
    --celltype_key celltype \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/data/SubMouseBrain_raw.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/brain \
    --batch_key BATCH \
    --celltype_key celltype \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/experiments/celltypist_annotaion/data/celltypist_immune.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/immune_celltypist \
    --batch_key batch \
    --celltype_key celltypist_label \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/experiments/celltypist_annotaion/data/celltypist_lung.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/lung_celltypist \
    --batch_key batch \
    --celltype_key celltypist_label \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/experiments/celltypist_annotaion/data/celltypist_pancreas.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/pancreas_celltypist \
    --batch_key tech \
    --celltype_key celltypist_label \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scSCI/experiments/ATAC/data/pbmc_multiome_ATAC.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/batch_effect/result/ATAC \
    --batch_key batch \
    --celltype_key celltype \
    --n_runs 5