from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LABELS = ["safe", "not safe"]


def normalize_label(raw_text):
    """Turns the model's raw text output into a clean label."""
    text = str(raw_text).strip().lower()

    if "not safe" in text or "unsafe" in text or "profane" in text:
        return "not safe"

    if "safe" in text:
        return "safe"

    return "unknown"


def compute_metrics(results):
    """Computes overall metrics when gold labels are available."""

    pairs = [
        (r["gold_label"], r["predicted_label"])
        for r in results
        if r.get("gold_label") in LABELS
    ]

    if not pairs:
        return None

    gold = [g for g, _ in pairs]
    pred = [p for _, p in pairs]

    return {
        "accuracy": round(
            accuracy_score(gold, pred),
            4
        ),

        "precision": round(
            precision_score(
                gold,
                pred,
                average="macro",
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                gold,
                pred,
                average="macro",
                zero_division=0
            ),
            4
        ),

        "f1": round(
            f1_score(
                gold,
                pred,
                average="macro",
                zero_division=0
            ),
            4
        ),
    }
