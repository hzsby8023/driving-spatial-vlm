
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.train_lora_smoke import (
    build_lora_model,
    build_processor,
)
from vla.models.driving_vla import (
    DrivingVLA,
    trajectory_loss,
)
from vla.data.nuscenes_vla_dataset import NuScenesVLADataset
from vla.training.smoke_vla import prepare_inputs
from vla.evaluation.planning_metrics import (
    planning_metrics,
    constant_velocity_baseline,
)


def loader(manifest, dataroot, shuffle):
    dataset = NuScenesVLADataset(manifest, dataroot)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )


def forward_sample(model, processor, sample, device):
    inputs = prepare_inputs(
        processor, sample["images"]
    ).to(device)

    history = sample["ego_history"].unsqueeze(0).to(device)
    target = sample["future_trajectory"].unsqueeze(0).to(device)
    history_times = (
        sample["history_time_offsets_sec"].unsqueeze(0).to(device)
    )
    future_times = (
        sample["future_time_offsets_sec"].unsqueeze(0).to(device)
    )

    pred = model(
        vlm_inputs=inputs,
        ego_history=history,
        history_times=history_times,
        future_times=future_times,
        train_backbone=True,
    )
    return pred, target


@torch.no_grad()
def evaluate(model, processor, val_loader, device, limit):
    model.eval()

    totals = {
        "ade": 0.0,
        "fde": 0.0,
        "cv_ade": 0.0,
        "cv_fde": 0.0,
    }
    count = 0

    for sample in val_loader:
        pred, target = forward_sample(
            model, processor, sample, device
        )
        result = planning_metrics(pred, target)

        cv = constant_velocity_baseline(
            sample["ego_history"].unsqueeze(0).to(device),
            sample["history_time_offsets_sec"].unsqueeze(0).to(device),
            sample["future_time_offsets_sec"].unsqueeze(0).to(device),
        )
        cv_result = planning_metrics(cv, target)

        totals["ade"] += result["ade"]
        totals["fde"] += result["fde"]
        totals["cv_ade"] += cv_result["ade"]
        totals["cv_fde"] += cv_result["fde"]
        count += 1

        if limit and count >= limit:
            break

    model.train()
    return {
        key: value / count
        for key, value in totals.items()
    }


def save_checkpoint(model, output_dir, step, metrics):
    path = Path(output_dir) / f"step_{step:05d}"
    path.mkdir(parents=True, exist_ok=True)

    # Save PEFT adapter, not the entire 4-bit base model.
    model.backbone.save_pretrained(path / "adapter")

    modules = [
        "ego_encoder",
        "visual_proj",
        "fusion",
        "trajectory_head",
    ]
    state = {
        name: {
            k: v.detach().cpu()
            for k, v in getattr(model, name).state_dict().items()
        }
        for name in modules
    }

    torch.save(state, path / "vla_heads.pt")

    with open(path / "metrics.json", "w") as f:
        json.dump(
            {"step": step, **metrics},
            f, indent=2
        )

    print("Checkpoint:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/datasets/nuscenes")
    parser.add_argument("--data-dir", default="data/vla/nuscenes_mini")
    parser.add_argument("--output-dir", default="outputs/vla_v0")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    assert torch.cuda.is_available()

    train_loader = loader(
        f"{args.data_dir}/train.jsonl",
        args.dataroot, True,
    )
    val_loader = loader(
        f"{args.data_dir}/val.jsonl",
        args.dataroot, False,
    )

    processor = build_processor(max_visual_tokens=64)
    backbone = build_lora_model()

    model = DrivingVLA(
        backbone=backbone,
        hidden_size=backbone.config.text_config.hidden_size,
        horizon=6,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()

    while step < args.max_steps:
        for sample in train_loader:
            optimizer.zero_grad(set_to_none=True)

            pred, target = forward_sample(
                model, processor, sample, device
            )

            loss, details = trajectory_loss(pred, target)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step + 1}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            step += 1

            print(
                f"step={step} "
                f"loss={loss.item():.4f} "
                f"xy={details['position']:.4f} "
                f"yaw={details['yaw']:.4f}",
                flush=True,
            )

            if step % args.eval_every == 0:
                metrics = evaluate(
                    model, processor, val_loader,
                    device, args.val_limit,
                )
                print("Validation:", metrics, flush=True)
                save_checkpoint(
                    model, args.output_dir, step, metrics
                )

            if step >= args.max_steps:
                break

    print("Training completed:", step)


if __name__ == "__main__":
    main()
