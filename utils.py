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
    Extracts predicted label and reason from model output.

    Expected model output:

    {
        "label": "safe",
        "reason": "The text does not contain profanity."
    }
    """

    text = str(raw_text).strip()

    # Try JSON first
    try:
        result = json.loads(text)

        label = result.get("label", "")
        reason = result.get("reason", "")

        return normalize_label(label), str(reason).strip()

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback if model did not return valid JSON
    label = normalize_label(text)

    return label, ""


def normalize_label(raw_text):
    """
    Converts model output into:
        safe
        not safe
        unknown
    """

    text = str(raw_text).strip().lower()

    if "not safe" in text or "unsafe" in text or "profane" in text:
        return "not safe"

    if "safe" in text:
        return "safe"

    return "unknown"


def compute_metrics(results):
    """
    Computes accuracy, macro precision, macro recall and macro F1.
    Metrics are calculated only when gold labels are available.
    """

    pairs = [
        (r["gold_label"], r["predicted_label"])
        for r in results
        if r.get("gold_label") in LABELS
    ]

    if not pairs:
        return None

    gold = [g for g, _ in pairs]
    pred = [p for _, p in pairs]

    accuracy = accuracy_score(gold, pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
