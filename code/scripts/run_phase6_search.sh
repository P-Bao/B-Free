#!/usr/bin/env bash
set -e

# ==============================================================================
# Phase 6 Search: So sánh Baseline vs Full Fine-Tune vs LoRA (25% data, 3 epochs)
# ==============================================================================

CONFIG="configs/bfree_dcs.yaml"
DATA_ROOT="./training_data"
TRAIN_CSV="./training_data/train_subset_25pct.csv"
VAL_CSV="./training_data/valid_list.csv"
RESULTS_DIR="./results/phase6_search"
mkdir -p ${RESULTS_DIR}

echo "=== [1/3] Huấn luyện Full Fine-Tuning ==="
python train_full_ft.py \
    --config ${CONFIG} \
    --train_csv ${TRAIN_CSV} \
    --val_csv ${VAL_CSV} \
    --data_root ${DATA_ROOT} \
    --output_dir "${RESULTS_DIR}/full_ft" \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5

echo "=== [2/3] Huấn luyện LoRA (Rank 16) ==="
python train_lora.py \
    --config ${CONFIG} \
    --train_csv ${TRAIN_CSV} \
    --val_csv ${VAL_CSV} \
    --data_root ${DATA_ROOT} \
    --output_dir "${RESULTS_DIR}/lora_r16" \
    --lora_rank 16 \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-4

echo "=== [3/3] Đánh giá Benchmark & Tổng hợp So sánh ==="
python eval/run_benchmarks.py \
    --config ${CONFIG} \
    --checkpoints \
        Baseline=./weights/BFREE_dino2reg4/model_epoch_best.pth \
        Full_FT="${RESULTS_DIR}/full_ft/best_model.pth" \
        LoRA_r16="${RESULTS_DIR}/lora_r16/best_model_lora.pth" \
    --benchmark_csvs \
        Synthbuster=./benchmarks/synthbuster.csv \
        GenImage=./benchmarks/genimage.csv \
    --data_root "./" \
    --output_csv "${RESULTS_DIR}/phase6_comparison_summary.csv"

echo "=== Phân tích Loss Dynamics ==="
python eval/analyze_loss.py \
    --log_csv "${RESULTS_DIR}/full_ft/train_log.csv" \
    --output_dir "${RESULTS_DIR}/full_ft_loss_analysis"

echo "[✓] Toàn bộ Phase 6 Search đã hoàn tất. Kết quả được lưu tại: ${RESULTS_DIR}"