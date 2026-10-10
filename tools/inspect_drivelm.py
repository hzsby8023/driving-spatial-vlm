from pathlib import Path
from pprint import pprint
from collections import Counter

from datasets.drivelm_dataset import DriveLMParser


DATA_PATH = Path(
    "/home/leo/projects/DriveLM/challenge/data/train_sample.json"
)


def main():
    print("=" * 60)
    print("DriveLM Dataset Inspection")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Parse raw DriveLM JSON into canonical samples
    # --------------------------------------------------

    parser = DriveLMParser(DATA_PATH)

    samples = parser.parse()

    print(f"\nTotal canonical samples: {len(samples)}")

    # --------------------------------------------------
    # 2. Inspect first canonical sample
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("First Canonical Sample")
    print("=" * 60)

    pprint(
        samples[0],
        depth=3,
    )

    # --------------------------------------------------
    # 3. Task distribution
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Task Distribution")
    print("=" * 60)

    task_counts = Counter(
        sample["task"]
        for sample in samples
    )

    for task, count in sorted(
        task_counts.items()
    ):
        ratio = (
            count
            / len(samples)
            * 100
        )

        print(
            f"{task:20s}: "
            f"{count:4d} "
            f"({ratio:6.2f}%)"
        )

    # --------------------------------------------------
    # 4. Sample ID check
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Sample ID Check")
    print("=" * 60)

    sample_ids = [
        sample["sample_id"]
        for sample in samples
    ]

    unique_ids = set(sample_ids)

    print(
        f"Total sample IDs : "
        f"{len(sample_ids)}"
    )

    print(
        f"Unique sample IDs: "
        f"{len(unique_ids)}"
    )

    if len(sample_ids) == len(unique_ids):
        print("Sample ID uniqueness: PASS")
    else:
        print("Sample ID uniqueness: FAIL")

    # --------------------------------------------------
    # 5. Show first 10 QA samples
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("First 10 QA Samples")
    print("=" * 60)

    for i, sample in enumerate(
        samples[:10]
    ):
        print(
            f"\n[{i}]"
        )

        print(
            "sample_id:",
            sample["sample_id"],
        )

        print(
            "task:",
            sample["task"],
        )

        print(
            "qa_index:",
            sample["qa_index"],
        )

        print(
            "question:",
            sample["question"],
        )

        print(
            "answer:",
            sample["answer"],
        )


if __name__ == "__main__":
    main()