"""
Post-processing and metrics.
"""

import json

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

LABELS = ["safe", "not safe"]


def parse_model_output(raw_text):
    """
    Extract label and reason from the model output.

    Expected format:

    {
        "label": "safe",
        "reason": "The text does not contain profanity."
    }
    """

    text = str(raw_text).strip()

    try:
        result = json.loads(text)

        label = normalize_label(
            result.get("label", "")
        )

        reason = str(
            result.get("reason", "")
        ).strip()

        return label, reason

    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback if the model does not return valid JSON
        return normalize_label(text), ""


def normalize_label(raw_text):
    """
    Converts model output into:
    safe / not safe / unknown
    """

    text = str(raw_text).strip().lower()

    if (
        "not safe" in text
        or "unsafe" in text
        or "profane" in text
    ):
        return "not safe"

    if "safe" in text:
        return "safe"

    return "unknown"


def compute_metrics(results):
    """
    Computes only:
    accuracy
    precision
    recall
    f1
    """

    pairs = [
        (
            r["gold_label"],
            r["predicted_label"]
        )
        for r in results
        if r.get("gold_label") in LABELS
    ]

    if not pairs:
        return None

    gold = [g for g, _ in pairs]
    pred = [p for _, p in pairs]

    accuracy = accuracy_score(
        gold,
        pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            gold,
            pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
