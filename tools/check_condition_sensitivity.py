
"""Diagnose first-action conditioning for Qwen3-VL Driving Action SFT.

Run from the project root (or Docker /workspace).
Uses only CAM_FRONT by default.
"""
import sys
from pathlib import Path
import argparse
import copy
import hashlib

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla.data.driving_action_sft import (
    DrivingActionDataset,
    make_inputs,
)
from vla.models.driving_action_vla import load_for_inference


def fingerprint(obj):
    """Create a short fingerprint to compare images or text."""
    if isinstance(obj, Image.Image):
        return hashlib.sha256(obj.tobytes()).hexdigest()[:12]

    return hashlib.sha256(
        str(obj).encode("utf-8")
    ).hexdigest()[:12]


def parts(example):
    """Return multimodal content in the user message."""
    return example["messages"][1]["content"]


def image_parts(example):
    """Return (index, image-part) pairs."""
    return [
        (i, x)
        for i, x in enumerate(parts(example))
        if x.get("type") == "image"
    ]


def status_text_index(example):
    """Locate the text containing speed, acceleration and ego history."""
    matches = [
        i
        for i, x in enumerate(parts(example))
        if (
            x.get("type") == "text"
            and "Current speed:" in x.get("text", "")
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one status text, got {len(matches)}"
        )

    return matches[0]


def swapped(base, donor, mode):
    """
    Construct a controlled input swap.

    image:
        Keep base status, replace base images with donor images.

    state:
        Keep base images, replace base vehicle-status text
        with donor vehicle-status text.
    """
    result = copy.deepcopy(base)

    base_content = parts(result)
    donor_content = parts(donor)

    if mode == "image":
        bi = image_parts(result)
        di = image_parts(donor)

        if len(bi) != len(di):
            raise RuntimeError(
                "Cannot swap images: different image counts"
            )

        for (bidx, _), (_, dpart) in zip(bi, di):
            base_content[bidx]["image"] = (
                dpart["image"].copy()
            )

    elif mode == "state":
        bidx = status_text_index(result)
        didx = status_text_index(donor)

        base_content[bidx]["text"] = (
            donor_content[didx]["text"]
        )

    elif mode != "none":
        raise ValueError(mode)

    return result


def describe(example, label):
    """Print the input fingerprints and ground-truth actions."""
    ims = image_parts(example)

    s = parts(example)[
        status_text_index(example)
    ]["text"]

    print(
        f"\n{label}: "
        f"GT={example['row']['action_tokens']}"
    )

    print(
        "  images:",
        [
            (
                i,
                im["image"].size,
                fingerprint(im["image"]),
            )
            for i, im in ims
        ],
    )

    print(
        "  status_sha:",
        fingerprint(s),
        "status:",
        s[:260],
    )


def first_logits(
    model,
    processor,
    action_ids,
    example,
    device,
):
    """
    Compute first-action logits using the generation prefix.

    No teacher forcing:
    The last prefix position predicts the first action.
    """
    batch = make_inputs(
        processor,
        example,
        device,
        train=False,
    )

    with torch.inference_mode():
        output = model(
            **batch,
            use_cache=False,
        )

        # Restrict the result to the 2048 action tokens.
        logits = (
            output.logits[0, -1, action_ids]
            .float()
            .cpu()
        )

    del output, batch

    return logits


def report(
    name,
    logits,
    gt,
    reference=None,
    k=6,
):
    """Print top-k predictions and distribution differences."""
    probs = torch.softmax(logits, dim=0)

    vals, idx = logits.topk(k)

    pred = int(idx[0])

    gt_rank = (
        int((logits > logits[gt]).sum().item()) + 1
    )

    print(
        f"\n{name}: "
        f"pred={pred}, "
        f"GT={gt}, "
        f"rank={gt_rank}, "
        f"P(GT)={float(probs[gt]):.5f}"
    )

    print(
        "  top:",
        [
            (int(i), round(float(v), 3))
            for i, v in zip(idx, vals)
        ],
    )

    if reference is not None:
        delta = (logits - reference).abs()

        p0 = torch.softmax(reference, dim=0)

        probability_l1 = (
            probs - p0
        ).abs().sum()

        print(
            "  vs reference:",
            "mean_abs_logit_delta=",
            round(float(delta.mean()), 6),
            "max_abs_logit_delta=",
            round(float(delta.max()), 6),
            "probability_L1=",
            round(float(probability_l1), 6),
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--data",
        required=True,
    )

    ap.add_argument(
        "--image-root",
        required=True,
    )

    ap.add_argument(
        "--frames-per-camera",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--max-edge",
        type=int,
        default=448,
    )

    ap.add_argument(
        "--cameras",
        nargs="+",
        default=["CAM_FRONT"],
    )

    ap.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )

    ap.add_argument(
        "--swap-donor",
        type=int,
        default=1,
    )

    args = ap.parse_args()

    # 1. Load exactly the same camera configuration
    #    for all samples.
    ds = DrivingActionDataset(
        args.data,
        args.image_root,
        frames_per_camera=args.frames_per_camera,
        max_edge=args.max_edge,
        cameras=args.cameras,
    )

    indices = sorted(
        set(args.indices + [args.swap_donor])
    )

    examples = {
        i: ds[i]
        for i in indices
    }

    print(
        "Cameras:",
        args.cameras,
        "frames per camera:",
        args.frames_per_camera,
    )

    # 2. Check whether image and status inputs differ.
    for i in args.indices:
        describe(
            examples[i],
            f"Sample {i}",
        )

    # 3. Load trained model.
    model, processor, action_ids = (
        load_for_inference(
            args.checkpoint,
            args.model,
        )
    )

    device = (
        model.get_input_embeddings().weight.device
    )

    print(
        "\nModel device:",
        device,
        "action vocab:",
        len(action_ids),
    )

    # 4. Original first-action logits.
    all_logits = {}

    for i in args.indices:
        all_logits[i] = first_logits(
            model,
            processor,
            action_ids,
            examples[i],
            device,
        )

    base_idx = args.indices[0]
    base = examples[base_idx]

    reference = all_logits[base_idx]

    print("\n========== ORIGINAL INPUTS ==========")

    for i in args.indices:
        gt = int(
            examples[i]["row"]["action_tokens"][0]
        )

        report(
            f"Original sample {i}",
            all_logits[i],
            gt,
            reference=(
                reference
                if i != base_idx
                else None
            ),
        )

    # 5. Controlled swaps.
    donor_idx = args.swap_donor

    if donor_idx == base_idx:
        raise ValueError(
            "--swap-donor must differ from "
            "the first --indices value"
        )

    donor = examples[donor_idx]

    print("\n========== CONDITION SWAPS ==========")

    for mode in ["image", "state"]:
        modified = swapped(
            base,
            donor,
            mode,
        )

        changed = first_logits(
            model,
            processor,
            action_ids,
            modified,
            device,
        )

        report(
            (
                f"Sample {base_idx} "
                f"with sample {donor_idx} {mode}"
            ),
            changed,
            int(
                base["row"]["action_tokens"][0]
            ),
            reference=reference,
        )

    print("\n========== FINISHED ==========")

    print(
        "NOTE: Swap results measure prediction "
        "sensitivity, not causality or accuracy improvement."
    )

    print(
        "If different inputs produce nearly identical "
        "logits, inspect prompt and image processing."
    )


if __name__ == "__main__":
    main()
