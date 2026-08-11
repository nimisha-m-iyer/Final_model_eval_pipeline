"""
Post-processing only. No pre-processing is needed beyond reading
record["text"], so nothing extra is added here for that.
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
    """Only computes anything if at least one record had a gold label."""
    pairs = [(r["gold_label"], r["predicted_label"]) for r in results if r.get("gold_label") in LABELS]
    if not pairs:
        return None

    gold = [g for g, _ in pairs]
    pred = [p for _, p in pairs]

    accuracy = accuracy_score(gold, pred)
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=LABELS).tolist()

    return {
        "accuracy": round(accuracy, 4),
        "per_class": {
            LABELS[i]: {
                "precision": round(precision[i], 4),
                "recall": round(recall[i], 4),
                "f1": round(f1[i], 4),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }
