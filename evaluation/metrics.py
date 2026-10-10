import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Simple normalization for diagnostic evaluation.
    """

    text = text.lower().strip()

    # Remove punctuation.
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text


def exact_match(
    prediction: str,
    ground_truth: str,
) -> float:

    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    return float(pred == gt)


def token_f1(
    prediction: str,
    ground_truth: str,
) -> float:

    pred_tokens = normalize_text(
        prediction
    ).split()

    gt_tokens = normalize_text(
        ground_truth
    ).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = (
        num_same / len(pred_tokens)
    )

    recall = (
        num_same / len(gt_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


def contains_match(
    prediction: str,
    ground_truth: str,
) -> float:

    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    return float(gt in pred)