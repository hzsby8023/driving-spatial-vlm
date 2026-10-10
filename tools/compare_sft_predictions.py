import json
import re
from collections import Counter
from pathlib import Path


BASELINE = Path(
    "outputs/baseline/"
    "qwen3_vl_2b_4bit_lowres_eval_v0.jsonl"
)

SFT = Path(
    "outputs/qlora_sft_v0/"
    "eval_v0_predictions.jsonl"
)


def load_predictions(path):
    with path.open(encoding="utf-8") as f:
        return {
            row["sample_id"]: row
            for line in f
            if (row := json.loads(line))
        }


def tokenize(text):
    return re.findall(
        r"\w+",
        text.lower()
    )


def token_f1(prediction, ground_truth):

    pred = Counter(tokenize(prediction))
    gt = Counter(tokenize(ground_truth))

    overlap = sum((pred & gt).values())

    if not pred or not gt:
        return float(not pred and not gt)

    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred.values())
    recall = overlap / sum(gt.values())

    return (
        2 * precision * recall
        / (precision + recall)
    )


def main():

    baseline = load_predictions(BASELINE)
    sft = load_predictions(SFT)

    if set(baseline) != set(sft):
        raise ValueError(
            "Baseline and SFT sample IDs differ."
        )

    comparisons = []

    for sample_id, before in baseline.items():

        after = sft[sample_id]

        if (
            before["ground_truth"]
            != after["ground_truth"]
        ):
            raise ValueError(
                f"Ground truth mismatch: {sample_id}"
            )

        gt = before["ground_truth"]

        before_f1 = token_f1(
            before["prediction"], gt
        )

        after_f1 = token_f1(
            after["prediction"], gt
        )

        comparisons.append({
            "sample_id": sample_id,
            "task": before["task"],
            "question": before["question"],
            "ground_truth": gt,
            "before": before["prediction"],
            "after": after["prediction"],
            "before_f1": before_f1,
            "after_f1": after_f1,
            "delta": after_f1 - before_f1,
        })

    comparisons.sort(
        key=lambda x: x["delta"],
        reverse=True
    )

    def show_case(item):

        print("=" * 80)
        print("Task:", item["task"])
        print("Sample:", item["sample_id"])
        print("Question:", item["question"])
        print("GT:", item["ground_truth"])
        print("Zero-shot:", item["before"])
        print("SFT:", item["after"])
        print(
            "Token F1:",
            f"{item['before_f1']:.3f}",
            "->",
            f"{item['after_f1']:.3f}",
        )

    print("\nTOP 5 IMPROVEMENTS")

    for item in comparisons[:5]:
        show_case(item)

    print("\nTOP 5 REGRESSIONS")

    for item in comparisons[-5:]:
        show_case(item)

    print("\nSUMMARY")

    print(
        "Improved:",
        sum(x["delta"] > 0 for x in comparisons)
    )

    print(
        "Regressed:",
        sum(x["delta"] < 0 for x in comparisons)
    )

    print(
        "Unchanged:",
        sum(x["delta"] == 0 for x in comparisons)
    )


if __name__ == "__main__":
    main()