bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/experiments/celltypist_annotaion/data/celltypist_lung.h5ad \
    --save_path /Group16T/common/ccuc/scQuartz/result/lung_celltypist \
    --batch_key batch \
    --celltype_key celltypist_label \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/common/ccuc/scQuartz/experiments/celltypist_annotaion/data/celltypist_pancreas.h5ad \
    --save_path /Group16T/common/ccuc/scQuartz/result/pancreas_celltypist \
    --batch_key tech \
    --celltype_key celltypist_label \
    --n_runs 5