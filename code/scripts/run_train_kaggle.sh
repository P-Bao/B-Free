#!/usr/bin/env bash
# K2 LoRA training launcher (train-only) — chạy train_lora.py qua accelerate.
#
# Dùng trên Kaggle (notebook 02 gọi qua cell !bash) hoặc local:
#   bash scripts/run_train_kaggle.sh \
#       --train_data_root /kaggle/input/datasets/.../b-free-training-data \
#       --dinov2_sd /kaggle/input/.../model.safetensors \
#       --ai_genbench_dir /kaggle/input/datasets/.../ai-genbench-v1
#
# Tất cả hyperparameters có default locked D1-D10 trong train_lora.py;
# pass-through args ở đây để ghi đè khi cần (arg tường minh, fail sớm nếu thiếu).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Offline env (Target B: RTX PRO 6000, không internet/HF hub) — setdefault để
# không ghi đè giá trị notebook đã set trước đó.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# cwd = code dir để `from utils...` / `from networks...` import đúng repo module.
cd "${CODE_DIR}"

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend NO \
    train_lora.py "$@"
