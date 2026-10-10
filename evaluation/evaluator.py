import json
from collections import defaultdict
from pathlib import Path

from evaluation.metrics import (
    exact_match,
    token_f1,
    contains_match,
)


class BaselineEvaluator:

    def __init__(self, prediction_path):
        self.prediction_path = Path(
            prediction_path
        )

    def load(self):

        records = []

        with self.prediction_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                records.append(
                    json.loads(line)
                )

        return records

    def evaluate_record(self, record):

        if record.get("status") != "success":
            return {
                "success": 0.0,
                "exact_match": 0.0,
                "token_f1": 0.0,
                "contains_match": 0.0,
            }

        prediction = record["prediction"]
        ground_truth = record["ground_truth"]

        return {
            "success": 1.0,

            "exact_match":
                exact_match(
                    prediction,
                    ground_truth,
                ),

            "token_f1":
                token_f1(
                    prediction,
                    ground_truth,
                ),

            "contains_match":
                contains_match(
                    prediction,
                    ground_truth,
                ),
        }

    @staticmethod
    def aggregate(results):

        if not results:
            return {}

        keys = results[0].keys()

        return {
            key: sum(
                result[key]
                for result in results
            ) / len(results)

            for key in keys
        }

    def evaluate(self):

        records = self.load()

        all_results = []

        task_results = defaultdict(list)

        latencies = []
        peak_memories = []

        # ------------------------------------
        # Per sample
        # ------------------------------------

        for record in records:

            metrics = self.evaluate_record(
                record
            )

            all_results.append(metrics)

            task = record.get(
                "task",
                "unknown",
            )

            task_results[task].append(
                metrics
            )

            if record.get("status") == "success":

                if "latency" in record:
                    latencies.append(
                        record["latency"]
                    )

                if "peak_memory_gb" in record:
                    peak_memories.append(
                        record["peak_memory_gb"]
                    )

        # ------------------------------------
        # Overall
        # ------------------------------------

        report = {
            "num_samples": len(records),

            "overall":
                self.aggregate(
                    all_results
                ),

            "per_task": {},
        }

        if latencies:
            report["runtime"] = {
                "mean_latency_sec":
                    sum(latencies)
                    / len(latencies),

                "max_latency_sec":
                    max(latencies),
            }

        if peak_memories:

            report.setdefault(
                "runtime",
                {},
            )

            report["runtime"][
                "peak_memory_gb"
            ] = max(peak_memories)

        # ------------------------------------
        # Per task
        # ------------------------------------

        for task, results in task_results.items():

            report["per_task"][task] = {
                "num_samples": len(results),

                **self.aggregate(
                    results
                ),
            }

        return report