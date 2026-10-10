#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${AUTODL_DATA_ROOT}/datasets/nuscenes}"
AUTOVLA_CODEBOOK="${AUTOVLA_CODEBOOK:-${AUTODL_DATA_ROOT}/external/AutoVLA/codebook_cache/agent_vocab.pkl}"
REQUIRE_CODEBOOK="${REQUIRE_CODEBOOK:-0}"
HF_HOME="${HF_HOME:-${AUTODL_DATA_ROOT}/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AUTODL_DATA_ROOT}/runs/m1_qwen3_vl_2b_qlora}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/data/m1/train_sft.jsonl}"
VAL_DATA="${VAL_DATA:-${REPO_ROOT}/data/m1/val_sft.jsonl}"
export AUTODL_DATA_ROOT HF_HOME MODEL_NAME ALLOW_DOWNLOAD

cd "${REPO_ROOT}"
mkdir -p "${AUTODL_DATA_ROOT}" "${HF_HOME}" "${OUTPUT_ROOT}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import importlib
import os
import shutil
import torch

required = ('transformers', 'accelerate', 'peft', 'bitsandbytes', 'PIL', 'numpy')
for name in required:
    importlib.import_module(name)
if not torch.cuda.is_available():
    raise SystemExit('ERROR: torch.cuda.is_available() is False')
total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}')
print(f'GPU memory: {total_gib:.1f} GiB')
if total_gib < 5:
    raise SystemExit('ERROR: less than 5 GiB GPU memory is unsupported for the M1 defaults')
if total_gib < 11:
    print('WARNING: very low VRAM; the validated 6 GiB path has almost no safety margin')
elif total_gib < 20:
    print('WARNING: low VRAM; keep CAM_FRONT, one frame, and 64 visual tokens')
free_gib = shutil.disk_usage(os.environ['AUTODL_DATA_ROOT']).free / 1024**3
print(f'Free persistent storage: {free_gib:.1f} GiB')
if free_gib < 20:
    raise SystemExit('ERROR: less than 20 GiB free on /root/autodl-tmp')
PY

if [[ "${ALLOW_DOWNLOAD}" != "1" ]]; then
  python - <<'PY'
import os
from transformers import AutoConfig
AutoConfig.from_pretrained(
    os.environ['MODEL_NAME'],
    local_files_only=True,
    trust_remote_code=False,
)
print(f"Cached model metadata found: {os.environ['MODEL_NAME']}")
PY
else
  echo "WARNING: ALLOW_DOWNLOAD=1; model assets may be downloaded during training"
fi

for path in "${TRAIN_DATA}" "${VAL_DATA}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 1; }
done
[[ -d "${NUSCENES_ROOT}" ]] || { echo "ERROR: missing nuScenes root ${NUSCENES_ROOT}" >&2; exit 1; }
if [[ -f "${AUTOVLA_CODEBOOK}" ]]; then
  echo "Trusted codebook found: ${AUTOVLA_CODEBOOK}"
elif [[ "${REQUIRE_CODEBOOK}" == "1" ]]; then
  echo "ERROR: missing trusted codebook ${AUTOVLA_CODEBOOK}" >&2
  exit 1
else
  echo "WARNING: codebook not found; training can continue, but ADE/FDE evaluation is unavailable"
fi

python tools/validate_driving_action_data.py \
  "${TRAIN_DATA}" "${VAL_DATA}" \
  --image-root "${NUSCENES_ROOT}" \
  --check-images

echo "Preflight passed."
echo "HF_HOME=${HF_HOME}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
