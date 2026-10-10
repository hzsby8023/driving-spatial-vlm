# training/train_lora.py

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader
from datasets.sft_dataset import DriveLMSFTDataset
from datasets.sft_collator import Qwen3VLSFTCollator

from training.train_lora_smoke import (
    DATA_PATH,
    IMAGE_ROOT,
    EVAL_MANIFEST,
    build_processor,
    build_lora_model,
)


def load_train_samples():

    samples = DriveLMParser(DATA_PATH).parse()

    with EVAL_MANIFEST.open(
        "r", encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    eval_ids = {
        item["sample_id"]
        for item in manifest["samples"]
    }

    train_samples = [
        sample
        for sample in samples
        if sample["sample_id"] not in eval_ids
    ]

    assert len(train_samples) > 0

    train_ids = {
        sample["sample_id"]
        for sample in train_samples
    }

    assert train_ids.isdisjoint(eval_ids)

    print("Training samples:", len(train_samples))
    print("Eval QA excluded:", len(eval_ids))

    return train_samples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/qlora_sft_v0",
    )

    args = parser.parse_args()

    if args.max_steps < 1 or args.grad_accum < 1:
        raise ValueError(
            "max-steps and grad-accum must be positive."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    # -----------------------------------
    # 1. Dataset
    # -----------------------------------

    processor = build_processor(
        max_visual_tokens=64
    )

    train_samples = load_train_samples()

    image_loader = MultiViewImageLoader(
        IMAGE_ROOT
    )

    # Keep all six camera views.
    dataset = DriveLMSFTDataset(
        samples=train_samples,
        image_loader=image_loader,
    )

    collator = Qwen3VLSFTCollator(
        processor=processor
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    # -----------------------------------
    # 2. Model
    # -----------------------------------

    model = build_lora_model()

    # Quantized model is already on GPU.
    model.train()

    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.lr,
        weight_decay=0.01,
    )

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()

    # -----------------------------------
    # 3. Training
    # -----------------------------------

    global_step = 0
    micro_step = 0
    running_loss = 0.0

    history = []

    print()
    print("=" * 60)
    print("Qwen3-VL Six-view QLoRA Training")
    print("=" * 60)

    print("Max optimizer steps:", args.max_steps)
    print("Gradient accumulation:", args.grad_accum)
    print("Learning rate:", args.lr)

    for batch in loader:

        batch = {
            key: (
                value.to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

        if "pixel_values" in batch:
            batch["pixel_values"] = (
                batch["pixel_values"].to(
                    dtype=compute_dtype
                )
            )

        # Forward
        outputs = model(**batch)

        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss: {loss.item()}"
            )

        running_loss += loss.detach().item()

        # Scale loss for gradient accumulation.
        scaled_loss = loss / args.grad_accum

        # Backward
        scaled_loss.backward()

        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            (
                p for p in model.parameters()
                if p.requires_grad
            ),
            max_norm=1.0,
        )

        # Parameter update
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        mean_loss = (
            running_loss / args.grad_accum
        )

        running_loss = 0.0

        peak_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        history.append({
            "step": global_step,
            "loss": mean_loss,
            "peak_memory_gb": peak_gb,
        })

        print(
            f"step={global_step:04d} "
            f"loss={mean_loss:.4f} "
            f"peak_vram={peak_gb:.3f}GB",
            flush=True,
        )

        if global_step >= args.max_steps:
            break

    if global_step != args.max_steps:
        raise RuntimeError(
            f"Only completed {global_step} optimizer steps. "
            f"Requested {args.max_steps}. "
            "Increase available training samples "
            "or reduce max-steps."
        )

    # -----------------------------------
    # 4. Save adapter and metadata
    # -----------------------------------

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

    run_info = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "training": "six_view_qlora",
        "quantization": "nf4_4bit",
        "max_visual_tokens_per_image": 64,
        "num_cameras": 6,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.lr,
        "seed": 42,
        "history": history,
    }

    with (
        output_dir / "training_log.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            run_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("QLoRA Training: COMPLETE")
    print("=" * 60)

    print("Optimizer steps:", global_step)
    print("Adapter:", output_dir)


if __name__ == "__main__":
    main()