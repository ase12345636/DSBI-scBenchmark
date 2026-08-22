bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/GSE115978.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/GSE115978/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch samples \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/GSE123139.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/GSE123139/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch Sample.Short.Name \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/GSE153935.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/GSE153935/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch orig.ident \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/GSE166555.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/GSE166555/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch sample_id \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/Zhengsorted.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/Zhengsorted/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch None \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/lung_full.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/lung_full/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch None \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Integrated_training/train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/Independent_testing/sapiens_full.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/sapiens_full/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch Data_source \
    --test_batch donor \
    --n_runs 5

bash run_script.sh \
    --train_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/caf.data/caf_train.h5ad \
    --test_path /Group16T/common/ccuc/scLightGAT/scLightGAT/scLightGAT_data/caf.data/caf_test.h5ad \
    --save_path /Group16T/common/ccuc/scBenchmark/celltype_annotation/result/CAF/ \
    --train_celltype Celltype_training \
    --test_celltype Celltype_training \
    --train_batch SampleID \
    --test_batch None \
    --n_runs 5
