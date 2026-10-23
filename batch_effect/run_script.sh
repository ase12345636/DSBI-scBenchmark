#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /Group16T/common/ccuc/miniconda3/etc/profile.d/conda.sh
conda activate sctools

batch_key="batch"
celltype_key="celltype"
n_runs=5

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_path)
            dataset_path="$2"
            shift 2
            ;;
        --save_path)
            save_path="$2"
            shift 2
            ;;
        --batch_key)
            batch_key="$2"
            shift 2
            ;;
        --celltype_key)
            celltype_key="$2"
            shift 2
            ;;
        --n_runs)
            n_runs="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -z "$dataset_path" ]; then
    echo "Error: --dataset_path is required."
    exit 1
fi

if [ -z "$save_path" ]; then
    echo "Error: --save_path is required."
    exit 1
fi

echo "Dataset: $dataset_path"
echo "Save path: $save_path"
echo "Batch key: $batch_key"
echo "Celltype key: $celltype_key"

echo "Running Raw (PCA) batch correction"
python3 raw/raw.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times 1

echo "Running Harmony batch correction"
python3 Harmony/Harmony.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

echo "Running scVi batch correction"
python3 scVi/scVi.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

echo "Running Scanorama batch correction"
python3 Scanorama/Scanorama.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

echo "Running scANVI batch correction"
python3 scANVI/scANVI.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"


echo "Running scCM batch correction"
python3 scCM/sccm.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

echo "Running CONCORD batch correction"
python3 CONCORD/CONCORD.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

echo "Running BioBatchNet batch correction"
python3 BioBatchNet/BioBatchNet.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

echo "Running FADVI batch correction"
python3 FADVI/FADVI.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate SCIDRL
echo "Running SCIDRL batch correction"
python3 SCIDRL/SCIDRL.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate Seurat
echo "Running Seurat batch correction"
python3 Seurat/Seurat.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate scCobra
echo "Running scCobra batch correction"
python3 scCobra/sccobra.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate scDML
echo "Running scDML batch correction"
python3 scDML/scDMl.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate SCALEX
echo "Running SCALEX batch correction"
python3 SCALEX/SCALEX.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate scDREAMER
echo "Running scDREAMER batch correction"
python3 scDREAMER/scdreamer.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate CLEAR
echo "Running CLEAR batch correction"
python3 CLEAR/clear.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate MetaQ
echo "Running MetaQ batch correction"
python3 MetaQ/metaq.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate scDecorr
echo "Running scDecorr batch correction"
python3 scDecorr/scDecorr.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate AIF
echo "Running AIF batch correction"
python3 AIF/AIF.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate scMEDAL
echo "Running scMEDAL batch correction"
python3 scMEDAL/scmedal.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate scDML
echo "Running scDML-cluster batch correction"
python3 scDML-cluster/scDMl.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"

conda activate SMNN
echo "Running SMNN supervised batch correction"
python3 SMNN/SMNN.py \
    --dataset_path "$dataset_path" \
    --save_path "$save_path" \
    --batch_key "$batch_key" \
    --celltype_key "$celltype_key" \
    --run_times "$n_runs"
