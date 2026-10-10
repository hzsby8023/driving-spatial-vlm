import argparse
import json
from pathlib import Path

from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader
from datasets.view_selector import select_camera
from models.qwen3_vl import Qwen3VLPredictor


ANNOTATION_PATH = (
    "/home/leo/projects/DriveLM/"
    "challenge/data/train_sample.json"
)

NUSCENES_ROOT = (
    "/home/leo/projects/DriveLM/"
    "challenge/llama_adapter_v2_multimodal7b/"
    "data/nuscenes"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/baseline/qwen3_vl_2b_zero_shot.jsonl",
    )

    parser.add_argument(
        "--eval-manifest",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # if (
    #     args.eval_manifest is not None
    #     and args.limit is not None
    # ):
    #     raise ValueError(
    #         "--limit and --eval-manifest "
    #         "should not be used together."
    #     )

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    dataset_parser = DriveLMParser(
        ANNOTATION_PATH
    )

    samples = dataset_parser.parse()

    print(f"Total samples: {len(samples)}")

    if args.eval_manifest is not None:

        with open(
            args.eval_manifest,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        eval_ids = {
            item["sample_id"]
            for item in manifest["samples"]
        }

        samples = [
            sample
            for sample in samples
            if sample["sample_id"] in eval_ids
        ]

        print(
            f"Loaded eval manifest: "
            f"{args.eval_manifest}"
        )

        print(
            f"Evaluation samples: "
            f"{len(samples)}"
        )
    if args.eval_manifest is not None:

        expected = manifest["num_samples"]

        if len(samples) != expected:
            raise RuntimeError(
                f"Eval manifest expects "
                f"{expected} samples, "
                f"but found {len(samples)}."
            )
    # if args.limit is not None:
    #     samples = samples[:args.limit]

    print(f"Running samples: {len(samples)}")

    image_loader = MultiViewImageLoader(
        NUSCENES_ROOT
    )

    # -----------------------------------------
    # Model
    # Load ONCE
    # -----------------------------------------

    predictor = Qwen3VLPredictor()

    # -----------------------------------------
    # Output
    # -----------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Inference loop
    # -----------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for idx, sample in enumerate(samples):

            question = sample["question"]
            ground_truth = sample["answer"]
            task = sample["task"]
            
            try:
                images = image_loader.load(
                    sample["images"]
                )

                result = predictor.predict_multiview(
                    images=images,
                    question=question,
                )

                record = {
                    # -----------------------------
                    # Data identity
                    # -----------------------------
                    "sample_id": sample["sample_id"],
                    "scene_token": sample["scene_token"],
                    "frame_token": sample["frame_token"],
                    "qa_index": sample["qa_index"],

                    # -----------------------------
                    # Task
                    # -----------------------------
                    "task": task,

                    # -----------------------------
                    # Input / output
                    # -----------------------------
                    "num_images": 6,
                    "question": question,
                    "prediction": result["prediction"],
                    "ground_truth": ground_truth,

                    # -----------------------------
                    # Token statistics
                    # -----------------------------
                    "input_tokens": result["input_tokens"],

                    "visual_tokens_per_image": result.get(
                        "visual_tokens_per_image"
                    ),

                    "total_visual_tokens": result.get(
                        "total_visual_tokens"
                    ),

                    "image_grid_thw": result.get(
                        "image_grid_thw"
                    ),

                    # -----------------------------
                    # Runtime
                    # -----------------------------
                    "latency": result["latency"],
                    "peak_memory_gb": result["peak_memory_gb"],

                    "status": "success",
                }

            except Exception as e:

                record = {
                    "sample_id": idx,
                    "task": task,
                    # "camera": camera,
                    "question": question,
                    "ground_truth": ground_truth,
                    "status": "failed",
                    "error": str(e),
                }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()

            print(
                f"[{idx + 1}/{len(samples)}] "
                f"6-view | "
                f"{record['status']}"
            )

            if record["status"] == "success":

                print(
                    "Visual tokens:",
                    record["visual_tokens_per_image"],
                )

                print(
                    "Total visual tokens:",
                    record["total_visual_tokens"],
                )

                print(
                    "Input tokens:",
                    record["input_tokens"],
                )

                print(
                    "Peak VRAM:",
                    f"{record['peak_memory_gb']:.2f} GB",
                )

                print(
                    "Latency:",
                    f"{record['latency']:.2f} s",
                )

                print(
                    "Pred:",
                    record["prediction"],
                )

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()