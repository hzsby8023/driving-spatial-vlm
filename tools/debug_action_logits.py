import argparse
import sys
from pathlib import Path

import torch

# `python tools/debug_action_logits.py` puts tools/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla.data.driving_action_sft import (
    DrivingActionDataset,
    make_inputs,
)
from vla.models.driving_action_vla import load_for_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frames-per-camera", type=int, default=1)
    parser.add_argument("--max-edge", type=int, default=448)
    args = parser.parse_args()

    model, processor, action_ids = load_for_inference(
        args.checkpoint,
        args.model,
    )

    device = model.get_input_embeddings().weight.device

    ds = DrivingActionDataset(
        args.data,
        args.image_root,
        args.frames_per_camera,
        args.max_edge,
    )

    ex = ds[args.index]
    batch = make_inputs(
        processor,
        ex,
        device,
        train=True,
    )

    model.eval()

    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
        )

    # Causal LM:
    # logits[t-1] predicts labels[t].
    labels = batch["labels"][0]
    target_positions = (
        (labels >= 0)
        .nonzero(as_tuple=True)[0]
        .tolist()
    )

    action_id_to_class = {
        token_id: i
        for i, token_id in enumerate(action_ids)
    }

    gt_actions = []
    pred_actions = []

    print("Loss:", float(outputs.loss))

    for pos in target_positions:
        target_id = int(labels[pos])

        # Exclude EOS / other non-action targets.
        if target_id not in action_id_to_class:
            continue

        scores = outputs.logits[0, pos - 1, action_ids]
        pred_class = int(scores.argmax())
        gt_class = action_id_to_class[target_id]

        gt_actions.append(gt_class)
        pred_actions.append(pred_class)

        if len(gt_actions) == 1:
            top = torch.topk(scores, k=10)

            print("First action GT:", gt_class)
            print(
                "First action Top10:",
                [
                    (int(i), round(float(v), 3))
                    for i, v in zip(
                        top.indices,
                        top.values,
                    )
                ],
            )

            gt_rank = int(
                (scores > scores[gt_class]).sum().item()
            ) + 1

            print("First action GT rank:", gt_rank)

            # outputs.logits[0, pos - 1] is the full-vocabulary distribution
            # that predicts this action. scores above is only the 2048-way slice.
            full_logits = outputs.logits[0, pos - 1].float()
            action_logits = full_logits[action_ids]
            full_probs = torch.softmax(full_logits, dim=-1)
            action_mass = full_probs[action_ids].sum()
            action_probs = torch.softmax(action_logits, dim=-1)
            print("Action probability mass:", action_mass.item())
            print(
                "P(action_0 | action vocab):",
                action_probs[0].item(),
            )
            print(
                "P(GT action | action vocab):",
                action_probs[gt_class].item(),
            )
            full_top = torch.topk(full_logits, k=10)
            print(
                "Full vocab top10:",
                [
                    (
                        processor.tokenizer.convert_ids_to_tokens(int(i)),
                        round(float(v), 3),
                    )
                    for i, v in zip(
                        full_top.indices,
                        full_top.values,
                    )
                ],
            )

    print("GT actions:", gt_actions)
    print("Teacher-forced predictions:", pred_actions)

    if gt_actions:
        acc = sum(
            a == b for a, b in zip(gt_actions, pred_actions)
        ) / len(gt_actions)

        print("Teacher-forced action accuracy:", acc)


if __name__ == "__main__":
    main()