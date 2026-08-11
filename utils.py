"""
Post-processing and metrics.
"""

import re

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


LABELS = [
    "safe",
    "not safe"
]


def normalize_label(raw_text):

    """
    Converts model output into:

        safe
        not safe
        unknown
    """

    text = str(
        raw_text
    ).strip().lower()

    if (
        "not safe" in text
        or "unsafe" in text
        or "profane" in text
    ):

        return "not safe"

    if "safe" in text:

        return "safe"

    return "unknown"


def extract_reason(raw_text):

    """
    Extracts the last explanation after 'Reason:'.

    This is useful when the model produces something like:

        Classification: Not Safe

        Reason: The word is offensive...

    Only the explanation is returned.
    """

    text = str(
        raw_text
    ).strip()

    matches = list(
        re.finditer(
            r"reason\s*:",
            text,
            flags=re.IGNORECASE
        )
    )

    if matches:

        reason = text[
            matches[-1].end():
        ].strip()

        if reason:

            return reason

    return text


def compute_metrics(results):

    """
    Computes:

        accuracy
        precision
        recall
        f1

    using macro averaging.
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

    gold = [
        g
        for g, _ in pairs
    ]

    pred = [
        p
        for _, p in pairs
    ]

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
        "accuracy": round(
            accuracy,
            4
        ),
        "precision": round(
            precision,
            4
        ),
        "recall": round(
            recall,
            4
        ),
        "f1": round(
            f1,
            4
        )
    }
