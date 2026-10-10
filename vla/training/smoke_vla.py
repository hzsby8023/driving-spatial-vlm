import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from training.train_lora_smoke import (
    build_lora_model,
    build_processor,
)
from vla.models.driving_vla import (
    DrivingVLA,
    trajectory_loss,
)


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def load_sample(manifest, dataroot):
    with open(manifest, encoding="utf-8") as f:
        sample = json.loads(next(f))

    images = []
    for camera in CAMERAS:
        image_path = (
            Path(dataroot)
            / sample["cameras"][camera]["path"]
        )
        with Image.open(image_path) as img:
            images.append(img.convert("RGB"))

    ego_history = torch.tensor(
        sample["ego_history"],
        dtype=torch.float32,
    ).unsqueeze(0)

    future_gt = torch.tensor(
        sample["future_trajectory"],
        dtype=torch.float32,
    ).unsqueeze(0)

    history_times = torch.tensor(
        sample["history_time_offsets_sec"],
        dtype=torch.float32,
    ).unsqueeze(0)

    future_times = torch.tensor(
        sample["future_time_offsets_sec"],
        dtype=torch.float32,
    ).unsqueeze(0)

    return (
        sample,
        images,
        ego_history,
        future_gt,
        history_times,
        future_times,
    )


def prepare_inputs(processor, images):
    content = []

    for camera, image in zip(CAMERAS, images):
        content.append({
            "type": "text",
            "text": f"{camera}:",
        })
        content.append({
            "type": "image",
            "image": image,
        })

    content.append({
        "type": "text",
        "text": (
            "Represent the current driving scene "
            "for future ego trajectory prediction."
        ),
    })

    messages = [{
        "role": "user",
        "content": content,
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return processor(
        text=[prompt],
        images=images,
        padding=True,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/vla/nuscenes_mini/train.jsonl",
    )
    parser.add_argument(
        "--dataroot",
        default="/datasets/nuscenes",
    )
    parser.add_argument(
        "--train-lora",
        action="store_true",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available()

    device = torch.device("cuda:0")

    processor = build_processor(
        max_visual_tokens=64
    )
    backbone = build_lora_model()

    hidden_size = (
        backbone.config.text_config.hidden_size
    )

    model = DrivingVLA(
        backbone=backbone,
        hidden_size=hidden_size,
        horizon=6,
    ).to(device)

    if args.train_lora:
        model.train()
    else:
        model.eval()
        model.ego_encoder.train()
        model.visual_proj.train()
        model.fusion.train()
        model.trajectory_head.train()

    (
        sample,
        images,
        ego_history,
        gt,
        history_times,
        future_times,
    ) = load_sample(
        args.manifest,
        args.dataroot,
    )

    inputs = prepare_inputs(processor, images)
    inputs = inputs.to(device)
    ego_history = ego_history.to(device)
    gt = gt.to(device)
    history_times = history_times.to(device)
    future_times = future_times.to(device)

    # First run: train newly added layers only.
    # Second run: also update the LoRA backbone.
    params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and (
            args.train_lora
            or not name.startswith("backbone.")
        )
    ]

    optimizer = torch.optim.AdamW(
        params,
        lr=1e-4,
    )

    optimizer.zero_grad(set_to_none=True)

    pred = model(
        vlm_inputs=inputs,
        ego_history=ego_history,
        history_times=history_times,
        future_times=future_times,
        train_backbone=args.train_lora,
    )

    loss, details = trajectory_loss(pred, gt)

    loss.backward()

    head_grad = (
        model.trajectory_head[-1]
        .weight.grad.norm().item()
    )

    lora_grads = [
        p.grad.norm().item()
        for name, p in model.backbone.named_parameters()
        if "lora_" in name and p.grad is not None
    ]

    optimizer.step()

    print("Sample:", sample["sample_token"])
    print("Ego History:", tuple(ego_history.shape))
    print("Future GT:", tuple(gt.shape))
    print("Prediction:", tuple(pred.shape))
    print("Loss:", float(loss.detach()))
    print("Loss details:", details)
    print("Head grad norm:", head_grad)
    print("LoRA grads:", len(lora_grads))
    print("LoRA max grad:", max(lora_grads, default=0.0))
    print(
        "Peak allocated GB:",
        torch.cuda.max_memory_allocated()
        / 1024**3,
    )


if __name__ == "__main__":
    main()