bash run_script.sh \
    --dataset_path /Group16T/raw_data/scCobra/Immune_ALL_human.h5ad \
    --save_path /Group16T/common/ccuc/MyWork/result/immune/ \
    --batch_key batch \
    --celltype_key final_annotation \
    --n_runs 5

bash run_script.sh \
    --dataset_path /Group16T/raw_data/scCobra/Lung_atlas_public.h5ad \
    --save_path /Group16T/common/ccuc/MyWork/result/lung/ \
    --batch_key batch \
    --celltype_key cell_type \
    --n_runs 5 

bash run_script.sh \
    --dataset_path /Group16T/raw_data/scCobra/human_pancreas_norm_complexBatch.h5ad \
    --save_path /Group16T/common/ccuc/MyWork/result/pancreas/ \
    --batch_key tech \
    --celltype_key celltype \
    --n_runs 5 

# # Perturbed datasets (with OC calculation)
# bash run_script.sh \
#     --dataset_path /Group16T/common/ccuc/data/Immune_GOBP_perturbed_CD4T.h5ad \
#     --save_path /Group16T/common/ccuc/Workspace/batch_effect/MyWork/result/immune_CD4T/ \
#     --batch_key batch \
#     --celltype_key celltype \
#     --n_runs 5 \
#     --compute_oc

# bash run_script.sh \
#     --dataset_path /Group16T/common/ccuc/data/Immune_GOBP_perturbed_CD14.h5ad \
#     --save_path /Group16T/common/ccuc/Workspace/batch_effect/MyWork/result/immune_CD14/ \
#     --batch_key batch \
#     --celltype_key celltype \
#     --n_runs 5 \
#     --compute_oc 

# bash run_script.sh \
#     --dataset_path /Group16T/common/ccuc/data/Immune_perturbed_CD4T.h5ad \
#     --save_path /Group16T/common/ccuc/Workspace/batch_effect/MyWork/result/immune_CD4T_batch/ \
#     --batch_key batch \
#     --celltype_key celltype \
#     --n_runs 5 \
#     --compute_oc 

# bash run_script.sh \
#     --dataset_path /Group16T/common/ccuc/data/Immune_perturbed_CD14.h5ad \
#     --save_path /Group16T/common/ccuc/Workspace/batch_effect/MyWork/result/immune_CD14_batch/ \
#     --batch_key batch \
#     --celltype_key celltype \
#     --n_runs 5 \
#     --compute_oc 

# # ATAC
# bash run_script.sh \
#     --dataset_path /Group16T/common/ccuc/data/pbmc_multiome_scLVBags_ready.h5ad \
#     --save_path /Group16T/common/ccuc/Workspace/batch_effect/MyWork/result/ATAC/ \
#     --batch_key batch \
#     --celltype_key celltype \
#     --n_runs 5 \
#     --ATAC 
