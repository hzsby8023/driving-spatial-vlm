# training/train_lora_smoke.py

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader
from datasets.sft_dataset import DriveLMSFTDataset
from datasets.sft_collator import Qwen3VLSFTCollator


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

DATA_PATH = Path(
    "/home/leo/projects/DriveLM/"
    "challenge/data/train_sample.json"
)

IMAGE_ROOT = Path(
    "/home/leo/projects/DriveLM/"
    "challenge/llama_adapter_v2_multimodal7b/data/nuscenes"
)

EVAL_MANIFEST = Path(
    "data/splits/eval_v0.json"
)


def build_processor(
    max_visual_tokens=64
):

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    processor.image_processor.size = {
        "longest_edge": max_visual_tokens * 32 * 32,
        "shortest_edge": 32 * 32,
    }

    return processor


def load_training_sample():

    parser = DriveLMParser(DATA_PATH)
    samples = parser.parse()

    # Exclude exact QA IDs in eval_v0.
    # NOTE: This does not prevent scene/frame
    # overlap. It is only for pipeline smoke test.
    with EVAL_MANIFEST.open(
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    eval_ids = {
        item["sample_id"]
        for item in manifest["samples"]
    }

    candidates = [
        sample
        for sample in samples
        if sample["sample_id"] not in eval_ids
    ]

    if not candidates:
        raise RuntimeError(
            "No training candidates found."
        )

    return candidates[:1]


def get_llm_lora_targets(model):

    suffixes = (
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
    )

    targets = []

    for name, module in model.named_modules():

        # Only language-model decoder layers.
        if not name.startswith(
            "model.language_model.layers."
        ):
            continue

        if not name.endswith(suffixes):
            continue

        if isinstance(module, torch.nn.Linear):
            targets.append(name)

    if not targets:
        raise RuntimeError(
            "No LLM attention LoRA targets found. "
            "Inspect model.named_modules()."
        )

    print(
        f"Found {len(targets)} LLM LoRA targets"
    )

    print("Examples:")
    for name in targets[:8]:
        print(" ", name)

    return targets


def build_lora_model():

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("Compute dtype:", compute_dtype)

    # ---------------------------------------
    # 1. 4-bit quantization configuration
    # ---------------------------------------

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # ---------------------------------------
    # 2. Load quantized Qwen3-VL
    # ---------------------------------------

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            dtype=compute_dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
        )
    )

    model.config.use_cache = False

    # ---------------------------------------
    # 3. Prepare for QLoRA training
    # ---------------------------------------

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
    )

    # ---------------------------------------
    # 4. Inject LoRA into LLM Attention
    # ---------------------------------------

    targets = get_llm_lora_targets(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config
    )

    # ---------------------------------------
    # 5. Verify trainable parameters
    # ---------------------------------------

    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable

    for name in trainable:
        assert "lora_" in name, name
        assert "language_model" in name, name

    model.print_trainable_parameters()

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="outputs/lora_smoke",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for training."
        )

    device = torch.device("cuda")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("=" * 60)
    print("Qwen3-VL LoRA SFT Smoke Test")
    print("=" * 60)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # ---------------------------------------
    # 1. Dataset and Collator
    # ---------------------------------------

    processor = build_processor()

    samples = load_training_sample()

    print(
        "Training sample ID:",
        samples[0]["sample_id"]
    )

    image_loader = MultiViewImageLoader(
        IMAGE_ROOT
    )

    dataset = DriveLMSFTDataset(
        samples=samples,
        image_loader=image_loader,
    )

    collator = Qwen3VLSFTCollator(
        processor=processor
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # ---------------------------------------
    # 2. Model + LoRA
    # ---------------------------------------

    model = build_lora_model()

    # Quantized model is already placed on GPU
    # by device_map={"": 0}.
    # Do not call model.to(device).

    # model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=1e-4,
        weight_decay=0.01,
    )

    # ---------------------------------------
    # 3. One training batch
    # ---------------------------------------

    batch = next(iter(loader))

    batch = {
        key: (
            value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }

    # Match vision compute dtype.
    if "pixel_values" in batch:
        batch["pixel_values"] = (
            batch["pixel_values"].to(
                dtype=torch.bfloat16
            )
        )

    print()
    print("Batch shapes:")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(
                f"{key:25s}: "
                f"{tuple(value.shape)}"
            )

    supervised_tokens = (
        batch["labels"] != -100
    ).sum().item()

    print(
        "Supervised tokens:",
        supervised_tokens
    )

    assert supervised_tokens > 0

    # ---------------------------------------
    # 4. Snapshot a LoRA B matrix
    # ---------------------------------------

    tracked_name = None
    tracked_param = None

    for name, parameter in model.named_parameters():

        if (
            parameter.requires_grad
            and "lora_B" in name
        ):
            tracked_name = name
            tracked_param = parameter
            break

    if tracked_param is None:
        raise RuntimeError(
            "No trainable LoRA B matrix found."
        )

    before = (
        tracked_param.detach()
        .float()
        .cpu()
        .clone()
    )

    # ---------------------------------------
    # 5. Forward
    # ---------------------------------------

    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)

    loss = outputs.loss

    print()
    print("Loss before update:", loss.item())

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Loss is NaN or Inf."
        )

    # ---------------------------------------
    # 6. Backward
    # ---------------------------------------

    loss.backward()

    grad = tracked_param.grad

    if grad is None:
        raise RuntimeError(
            "Tracked LoRA parameter has no gradient."
        )

    grad_norm = grad.detach().float().norm().item()

    print(
        "Tracked LoRA B gradient norm:",
        grad_norm
    )

    if grad_norm <= 0:
        raise RuntimeError(
            "Tracked LoRA B gradient is zero."
        )

    # ---------------------------------------
    # 7. Optimizer update
    # ---------------------------------------

    torch.nn.utils.clip_grad_norm_(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        max_norm=1.0,
    )

    optimizer.step()

    after = (
        tracked_param.detach()
        .float()
        .cpu()
        .clone()
    )

    max_change = (
        after - before
    ).abs().max().item()

    print(
        "Tracked parameter:",
        tracked_name
    )

    print(
        "Max parameter change:",
        max_change
    )

    if max_change <= 0:
        raise RuntimeError(
            "LoRA weights did not update."
        )

    # ---------------------------------------
    # 8. Memory statistics
    # ---------------------------------------

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(
        "Peak allocated VRAM (GB):",
        round(peak_gb, 3)
    )

    # ---------------------------------------
    # 9. Save adapter
    # ---------------------------------------

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        str(output_dir)
    )

    processor.save_pretrained(
        str(output_dir)
    )

    print()
    print("=" * 60)
    print("LoRA SFT Smoke Test: PASS")
    print("=" * 60)

    print(
        "Adapter saved to:",
        output_dir
    )


if __name__ == "__main__":
    main()