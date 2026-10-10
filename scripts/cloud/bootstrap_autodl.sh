#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements-cloud.txt"

mkdir -p \
  "${AUTODL_DATA_ROOT}/datasets/nuscenes" \
  "${AUTODL_DATA_ROOT}/external/AutoVLA/codebook_cache" \
  "${AUTODL_DATA_ROOT}/hf-cache" \
  "${AUTODL_DATA_ROOT}/runs"

echo "Cloud environment prepared."
echo "Repository: ${REPO_ROOT}"
echo "Persistent data root: ${AUTODL_DATA_ROOT}"
echo "Next: upload nuScenes and agent_vocab.pkl, then run scripts/cloud/preflight.sh"
