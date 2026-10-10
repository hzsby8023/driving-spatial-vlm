
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/vla/nuscenes_mini/train.jsonl",
    )
    parser.add_argument(
        "--index", type=int, default=0
    )
    parser.add_argument(
        "--output",
        default="outputs/vla/trajectory_preview.png",
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    sample = records[args.index]

    history = np.array(sample["ego_history"])
    future = np.array(sample["future_trajectory"])

    # Plot X forward upward and Y left to the left.
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.plot(
        -history[:, 1], history[:, 0],
        "o--", label="Ego History",
    )

    future_with_origin = np.vstack([
        np.zeros((1, 3)), future
    ])

    ax.plot(
        -future_with_origin[:, 1],
        future_with_origin[:, 0],
        "o-", label="Future GT",
    )

    ax.scatter(
        [0], [0],
        marker="s", s=100,
        label="Current Ego",
        zorder=5,
    )

    for i, point in enumerate(future):
        ax.annotate(
            f"{sample['future_time_offsets_sec'][i]:.1f}s",
            (-point[1], point[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Right (m)")
    ax.set_ylabel("Forward (m)")
    ax.set_title(
        f"{sample['scene_name']}\n"
        f"{sample['sample_token'][:12]}"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output = Path(args.output)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(
        output, dpi=160, bbox_inches="tight"
    )
    plt.close(fig)

    print("Saved:", output)
    print("History:", history.shape)
    print("Future GT:", future.shape)
    print("Future time offsets:",
          sample["future_time_offsets_sec"])


if __name__ == "__main__":
    main()
