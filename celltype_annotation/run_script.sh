#!/usr/bin/env bash

source /Group16T/common/ccuc/miniconda3_gb10/etc/profile.d/conda.sh

train_celltype="celltype"
test_celltype="celltype"
train_batch="batch"
test_batch="batch"
n_runs=1
base_seed=""
save_models=""
scgpt_model_dir=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --train_path)
            train_path="$2"
            shift 2
            ;;
        --test_path)
            test_path="$2"
            shift 2
            ;;
        --save_path)
            save_path="$2"
            shift 2
            ;;
        --train_celltype)
            train_celltype="$2"
            shift 2
            ;;
        --test_celltype)
            test_celltype="$2"
            shift 2
            ;;
        --train_batch)
            train_batch="$2"
            shift 2
            ;;
        --test_batch)
            test_batch="$2"
            shift 2
            ;;
        --n_runs|--run_times)
            n_runs="$2"
            shift 2
            ;;
        --base_seed)
            base_seed="$2"
            shift 2
            ;;
        --save_models)
            save_models="--save_models"
            shift
            ;;
        --scgpt_model_dir)
            scgpt_model_dir="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown parameter: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${train_path:-}" ]]; then
    echo "Error: --train_path is required." >&2
    exit 1
fi
if [[ -z "${test_path:-}" ]]; then
    echo "Error: --test_path is required." >&2
    exit 1
fi
if [[ -z "${save_path:-}" ]]; then
    echo "Error: --save_path is required." >&2
    exit 1
fi
if ! [[ "$n_runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --n_runs must be a positive integer." >&2
    exit 1
fi

common_args=(
    --train_path "$train_path"
    --test_path "$test_path"
    --save_path "$save_path"
    --train_celltype "$train_celltype"
    --test_celltype "$test_celltype"
    --train_batch "$train_batch"
    --test_batch "$test_batch"
    --n_runs "$n_runs"
)
if [[ -n "$base_seed" ]]; then
    common_args+=(--base_seed "$base_seed")
fi
if [[ -n "$save_models" ]]; then
    common_args+=(--save_models)
fi

echo "Train dataset: $train_path"
echo "Test dataset: $test_path"
echo "Save path: $save_path"
echo "Train celltype: $train_celltype"
echo "Test celltype: $test_celltype"
echo "Train batch: $train_batch"
echo "Test batch: $test_batch"
echo "Runs: $n_runs"

failed_methods=()

run_method() {
    local method_name="$1"
    local script_path="$2"
    shift 2
    echo
    echo "Running $method_name cell type annotation"
    if ! python3 "$script_path" "${common_args[@]}" "$@"; then
        echo "Error: $method_name failed; continuing with remaining methods." >&2
        failed_methods+=("$method_name")
    fi
}

if conda activate Celltypist; then
    run_method "Celltypist" "Celltypist/Celltypist.py"
else
    failed_methods+=("Celltypist(environment)")
fi

if conda activate CHETAH; then
    run_method "CHETAH" "CHETAH/CHETAH.py"
else
    failed_methods+=("CHETAH(environment)")
fi

if conda activate Seurat; then
    run_method "Seurat" "Seurat/Seurat.py"
else
    failed_methods+=("Seurat(environment)")
fi

if conda activate SingleR; then
    run_method "SingleR" "SingleR/SingleR.py"
else
    failed_methods+=("SingleR(environment)")
fi

if conda activate scGPT; then
    if [[ -z "$scgpt_model_dir" ]]; then
        scgpt_model_dir="scGPT/models/scGPT_human"
    fi
    if [[ ! -f "$scgpt_model_dir/best_model.pt" || \
          ! -f "$scgpt_model_dir/vocab.json" || \
          ! -f "$scgpt_model_dir/args.json" ]]; then
        python3 scGPT/download_checkpoint.py --output_dir "$scgpt_model_dir"
    fi
    run_method "scGPT" "scGPT/scGPT.py" \
        --scgpt_model_dir "$scgpt_model_dir"
else
    failed_methods+=("scGPT(environment)")
fi

if [[ "${#failed_methods[@]}" -gt 0 ]]; then
    echo
    echo "Failed methods: ${failed_methods[*]}" >&2
    exit 1
fi

echo
echo "All annotation benchmarks completed successfully."
