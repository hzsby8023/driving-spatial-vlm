#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${AUTODL_DATA_ROOT}/datasets/nuscenes}"
HF_HOME="${HF_HOME:-${AUTODL_DATA_ROOT}/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AUTODL_DATA_ROOT}/runs/m1_qwen3_vl_2b_qlora}"
RESUME_FROM="${RESUME_FROM:-}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO_ROOT}"
"${REPO_ROOT}/scripts/cloud/preflight.sh"

args=(
  python -m vla.training.train_driving_action_sft
  --config configs/m1_qwen3_vl_2b_qlora.json
  --image-root "${NUSCENES_ROOT}"
  --output "${OUTPUT_ROOT}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  args+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
  args+=(--allow-download)
fi

echo "Starting M1 training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
"${args[@]}"
