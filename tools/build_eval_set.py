import json
import random
from collections import Counter
from pathlib import Path

from datasets.drivelm_dataset import DriveLMParser


DATA_PATH = Path(
    "/home/leo/projects/DriveLM/challenge/data/train_sample.json"
)

OUTPUT_PATH = Path(
    "data/splits/eval_v0.json"
)

SEED = 42

TASK_TARGETS = {
    "perception": 40,
    "prediction": 30,
    "planning": 21,
    "behavior": 9,
}


def main():
    random.seed(SEED)

    parser = DriveLMParser(DATA_PATH)
    samples = parser.parse()

    # --------------------------------------------
    # Group samples by task
    # --------------------------------------------

    task_samples = {}

    for sample in samples:
        task = sample["task"]

        task_samples.setdefault(
            task,
            [],
        ).append(sample)

    # --------------------------------------------
    # Stratified sampling
    # --------------------------------------------

    eval_samples = []

    for task, target_count in TASK_TARGETS.items():

        candidates = task_samples.get(
            task,
            [],
        )

        if not candidates:
            print(
                f"[WARNING] No samples for task: {task}"
            )
            continue

        count = min(
            target_count,
            len(candidates),
        )

        selected = random.sample(
            candidates,
            count,
        )

        eval_samples.extend(selected)

        print(
            f"{task:20s}: "
            f"{len(candidates):4d} available -> "
            f"{count:3d} selected"
        )

    # --------------------------------------------
    # Shuffle final evaluation set
    # --------------------------------------------

    random.shuffle(eval_samples)

    # --------------------------------------------
    # Save only stable identity + metadata
    #
    # Do NOT duplicate image/question/answer here.
    # Source of truth remains DriveLMParser.
    # --------------------------------------------

    manifest = {
        "name": "eval_v0",
        "seed": SEED,
        "num_samples": len(eval_samples),

        "samples": [
            {
                "sample_id":
                    sample["sample_id"],

                "scene_token":
                    sample["scene_token"],

                "frame_token":
                    sample["frame_token"],

                "task":
                    sample["task"],

                "qa_index":
                    sample["qa_index"],
            }
            for sample in eval_samples
        ],
    }

    # --------------------------------------------
    # Output directory
    # --------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------
    # Verify distribution
    # --------------------------------------------

    counts = Counter(
        sample["task"]
        for sample in eval_samples
    )

    unique_sample_ids = {
        sample["sample_id"]
        for sample in eval_samples
    }

    print()
    print("=" * 60)
    print("Evaluation Set V0")
    print("=" * 60)

    print(
        f"Total samples: {len(eval_samples)}"
    )

    print(
        f"Unique IDs   : {len(unique_sample_ids)}"
    )

    print()

    for task, count in sorted(counts.items()):
        print(
            f"{task:20s}: {count:3d}"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()