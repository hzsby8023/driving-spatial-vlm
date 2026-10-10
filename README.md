# Driving Spatial VLM — M1

M1 trains Qwen3-VL-2B-Instruct with 4-bit QLoRA to predict exactly ten action
tokens from a 2048-token vocabulary. The current action codec is
`greedy_endpoint_proxy_v1`; it uses the AutoVLA codebook for rollout but is not
the official K-Disk encoder.

## Preflight

Build the image, mount the repository at `/workspace`, nuScenes at
`/datasets/nuscenes`, the trusted AutoVLA codebook directory at `/external`, and
the Hugging Face cache at `/hf-cache`. Then validate data before renting a long
GPU run:

```bash
python tools/validate_driving_action_data.py \
  outputs/autovla_codebook/train_sft.jsonl \
  outputs/autovla_codebook/val_sft.jsonl \
  --image-root /datasets/nuscenes --check-images
```

The validator rejects malformed rows, duplicate samples, train/validation scene
overlap, future-GT navigation leakage, invalid action IDs, and incompatible
codec metadata.

## Train and resume

```bash
python -m vla.training.train_driving_action_sft \
  --config configs/m1_qwen3_vl_2b_qlora.json
```

Each checkpoint contains the trainable adapter, processor, optimizer, scheduler,
epoch/batch/global-step position, and Python/NumPy/Torch/CUDA RNG state. Resume
from the path recorded in `outputs/m1_qwen3_vl_2b_qlora/checkpoints/latest.json`:

```bash
python -m vla.training.train_driving_action_sft \
  --config configs/m1_qwen3_vl_2b_qlora.json \
  --resume-from outputs/m1_qwen3_vl_2b_qlora/checkpoints/step_XXXXXXXX_epoch_XXX_batch_XXXXXXXX
```

Metrics are appended to `metrics.jsonl`; the fully resolved configuration is
saved as `resolved_config.json`.

## AutoDL

Choose a PyTorch 2.5.1 / CUDA 12.4 base image. AutoDL container instances do
not need Docker inside the instance. After cloning the repository:

```bash
bash scripts/cloud/bootstrap_autodl.sh
bash scripts/cloud/preflight.sh
bash scripts/cloud/train_m1.sh
```

Persistent assets default to `/root/autodl-tmp`: nuScenes under
`datasets/nuscenes`, the trusted codebook at
`external/AutoVLA/codebook_cache/agent_vocab.pkl`, Hugging Face cache under
`hf-cache`, and checkpoints under `runs`. Override any path with
`AUTODL_DATA_ROOT`, `NUSCENES_ROOT`, `AUTOVLA_CODEBOOK`, `HF_HOME`, or
`OUTPUT_ROOT`. Set `ALLOW_DOWNLOAD=1` only when the base model is not already in
the Hugging Face cache. The codebook is optional for training because the M1
JSONL already contains action tokens; use `REQUIRE_CODEBOOK=1` before trajectory
evaluation. Resume with `RESUME_FROM=/absolute/checkpoint/path`.

## Free-generation evaluation

Teacher-forced validation loss is only a training diagnostic. Report deployment
quality with constrained free generation plus valid rate, ADE, FDE, and token
accuracy:

```bash
python -m vla.evaluation.eval_driving_action_sft \
  --config configs/m1_qwen3_vl_2b_qlora.json \
  --checkpoint outputs/m1_qwen3_vl_2b_qlora/final \
  --output outputs/m1_qwen3_vl_2b_qlora/free_generation_val.json
```
