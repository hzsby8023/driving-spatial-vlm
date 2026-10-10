import argparse
import json

from evaluation.evaluator import (
    BaselineEvaluator,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    evaluator = BaselineEvaluator(
        args.predictions
    )

    report = evaluator.evaluate()

    print()
    print("=" * 60)
    print("Driving VLM Baseline Evaluation")
    print("=" * 60)

    print(
        f"Samples: "
        f"{report['num_samples']}"
    )

    print()

    print("Overall")

    for key, value in (
        report["overall"].items()
    ):
        print(
            f"  {key:20s}: "
            f"{value:.4f}"
        )

    if "runtime" in report:

        print()
        print("Runtime")

        for key, value in (
            report["runtime"].items()
        ):
            print(
                f"  {key:20s}: "
                f"{value:.4f}"
            )

    print()
    print("Per Task")

    for task, metrics in (
        report["per_task"].items()
    ):

        print(
            f"\n  [{task}] "
            f"n={metrics['num_samples']}"
        )

        for key, value in metrics.items():

            if key == "num_samples":
                continue

            print(
                f"    {key:18s}: "
                f"{value:.4f}"
            )

    print()
    print("=" * 60)

    if args.output:

        with open(
            args.output,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved report: {args.output}"
        )


if __name__ == "__main__":
    main()