# scripts/inference/run_qlora_eval.py

import argparse
import json
import time
from pathlib import Path

import torch

from transformers import (
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import PeftModel

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader
from datasets.sft_dataset import DriveLMSFTDataset

from training.train_lora_smoke import (
    MODEL_NAME,
    DATA_PATH,
    IMAGE_ROOT,
    EVAL_MANIFEST,
    build_processor,
)


def load_eval_samples(manifest_path):

    samples = DriveLMParser(DATA_PATH).parse()

    by_id = {
        sample["sample_id"]: sample
        for sample in samples
    }

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    ids = [
        item["sample_id"]
        for item in manifest["samples"]
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate sample IDs in eval manifest."
        )

    missing = [
        sid for sid in ids
        if sid not in by_id
    ]

    if missing:
        raise ValueError(
            f"Missing eval sample IDs: {missing[:5]}"
        )

    return [
        by_id[sid]
        for sid in ids
    ]


def load_model(adapter_path=None):

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            dtype=compute_dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
    )

    if adapter_path is not None:

        print(
            "Loading LoRA Adapter:",
            adapter_path
        )

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()

    return model, compute_dtype


@torch.inference_mode()
def predict(
    model,
    processor,
    sample,
    compute_dtype,
    max_new_tokens,
):

    # IMPORTANT:
    # Use only the user message.
    # Never feed the ground-truth assistant answer
    # into the model during evaluation.

    user_messages = sample["messages"][:1]

    inputs = processor.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = torch.device("cuda:0")

    inputs = {
        key: (
            value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = (
            inputs["pixel_values"].to(
                dtype=compute_dtype
            )
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    torch.cuda.synchronize()

    latency = time.perf_counter() - start

    # Decoder-only generation returns:
    # [prompt tokens, newly generated tokens]

    new_tokens = generated[
        0,
        input_length:
    ]

    prediction = (
        processor.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
    )

    grids = inputs["image_grid_thw"]

    merge_size = (
        processor.image_processor.merge_size
    )

    visual_tokens_per_image = (
        grids.prod(dim=1)
        // (merge_size ** 2)
    ).tolist()

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    return {
        "prediction": prediction,
        "input_tokens": input_length,
        "visual_tokens_per_image": (
            visual_tokens_per_image
        ),
        "total_visual_tokens": sum(
            visual_tokens_per_image
        ),
        "image_grid_thw": grids.tolist(),
        "latency": latency,
        "peak_memory_gb": peak_memory_gb,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA Adapter directory.",
    )

    parser.add_argument(
        "--eval-manifest",
        type=str,
        default=str(EVAL_MANIFEST),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    output_path = Path(args.output)

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------
    # 1. Dataset
    # -----------------------------------

    samples = load_eval_samples(
        args.eval_manifest
    )

    if args.limit is not None:
        samples = samples[:args.limit]

    image_loader = MultiViewImageLoader(
        IMAGE_ROOT
    )

    dataset = DriveLMSFTDataset(
        samples=samples,
        image_loader=image_loader,
    )

    # Exactly the same processor configuration
    # as the low-resolution QLoRA training.

    processor = build_processor(
        max_visual_tokens=64
    )

    # -----------------------------------
    # 2. Model
    # -----------------------------------

    model, compute_dtype = load_model(
        adapter_path=args.adapter
    )

    mode = (
        "SFT Adapter"
        if args.adapter
        else "Zero-shot"
    )

    print()
    print("=" * 60)
    print("Qwen3-VL Evaluation")
    print("=" * 60)

    print("Mode:", mode)
    print("Samples:", len(dataset))
    print("Output:", output_path)

    # -----------------------------------
    # 3. Inference
    # -----------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for index in range(len(dataset)):

            sample = dataset[index]

            result = predict(
                model=model,
                processor=processor,
                sample=sample,
                compute_dtype=compute_dtype,
                max_new_tokens=args.max_new_tokens,
            )

            record = {
                "sample_id": sample["sample_id"],
                "scene_token": sample["scene_token"],
                "frame_token": sample["frame_token"],
                "task": sample["task"],
                "question": sample["question"],
                "prediction": result["prediction"],
                "ground_truth": sample["answer"],
                "num_images": len(sample["images"]),
                "input_tokens": result["input_tokens"],
                "visual_tokens_per_image": (
                    result["visual_tokens_per_image"]
                ),
                "total_visual_tokens": (
                    result["total_visual_tokens"]
                ),
                "image_grid_thw": (
                    result["image_grid_thw"]
                ),
                "latency": result["latency"],
                "peak_memory_gb": (
                    result["peak_memory_gb"]
                ),
                "status": "success",
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )

            f.flush()

            print(
                f"[{index + 1}/{len(dataset)}] "
                f"task={sample['task']} "
                f"visual_tokens="
                f"{result['total_visual_tokens']} "
                f"latency={result['latency']:.2f}s",
                flush=True,
            )

    print()
    print("=" * 60)
    print("Evaluation Inference: COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()