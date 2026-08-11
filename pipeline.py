"""
======================================================================
 MINIMAL LLM EVALUATION PIPELINE

 Input:  a list of dicts
         [{"id": "1", "text": "...", "label": "safe", "language": "ml"}]

 Output:
         - predictions shown in the editor
         - predicted label
         - reason
         - metrics
         - CSV containing:
           id, predicted_label, reason, language

 The model is loaded exactly ONCE inside evaluate() and reused for every
 record / every batch.
======================================================================
"""

import csv
import os

from models import gemma, qwen, aya, llama
from utils import normalize_label, compute_metrics


# Add a new model by adding one line here
MODEL_MODULES = {
    "gemma": gemma,
    "qwen": qwen,
    "aya": aya,
    "llama": llama,
}


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert multilingual profanity detection system."
)

DEFAULT_USER_TEMPLATE = (
    "Classify the following text as 'safe' or 'not safe'. "
    "Then provide a complete and specific reason for your classification. "
    "The reason must explain the meaning of any potentially offensive word "
    "and why it is considered safe or not safe in the given language. "
    "Do not leave the reason incomplete.\n\n"
    "Text: {text}"
)


def _pick_module(model_path, model_type=None):

    if model_type:

        module = MODEL_MODULES.get(
            model_type.lower()
        )

        if module is None:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Use one of {list(MODEL_MODULES.keys())}"
            )

        return module

    name = model_path.lower()

    for key, module in MODEL_MODULES.items():

        if key in name:
            return module

    raise ValueError(
        f"Could not detect model type from '{model_path}'. "
        f"Either include one of {list(MODEL_MODULES.keys())} in the path, "
        f"or pass model_config['model_type'] explicitly."
    )


def _build_messages(text, prompt):

    return [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": prompt.format(text=text)
        },
    ]


def _extract_reason(raw_output):

    """
    Extract only the final explanation from the model output.

    Example:

    Classification: Not Safe

    Reason: The word is offensive...

    becomes:

    The word is offensive...
    """

    text = str(raw_output).strip()

    # Find the last occurrence of "reason:"
    lower_text = text.lower()

    position = lower_text.rfind("reason:")

    if position != -1:

        reason = text[
            position + len("reason:")
        ].strip()

        if reason:
            return reason

    # If the model did not use "Reason:",
    # return the complete model output.
    return text


def _save_csv(results, output_csv):

    fieldnames = [
        "id",
        "predicted_label",
        "reason",
        "language"
    ]

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for r in results:

            writer.writerow({
                "id": r.get("id"),
                "predicted_label": r.get("predicted_label"),
                "reason": r.get("reason"),
                "language": r.get("language")
            })

    print(
        f"[pipeline] predictions saved -> {output_csv}"
    )


def evaluate(
    records,
    model_config,
    prompt=None,
    mode="sequence",
    batch_size=8,
    output_csv=None
):
    """
    records:
        List of dictionaries.

        Example:
        [
            {
                "id": "1",
                "text": "some text",
                "label": "safe",
                "language": "ml"
            }
        ]

    model_config:
        {
            "model_path": "google/gemma-3-4b-it",
            "model_type": "gemma",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 100
        }

    prompt:
        Prompt supplied directly from the editor.

    mode:
        "sequence" or "batch"

    batch_size:
        Used only when mode == "batch"

    output_csv:
        CSV output path
    """

    # ------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------

    if prompt is None:
        prompt = DEFAULT_USER_TEMPLATE

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model_path = model_config["model_path"]

    module = _pick_module(
        model_path,
        model_config.get("model_type")
    )

    print(
        f"[pipeline] loading model from: {model_path}"
    )

    model, tokenizer = module.load(
        model_path,
        model_config.get(
            "torch_dtype",
            "bfloat16"
        ),
        model_config.get(
            "device_map",
            "auto"
        ),
    )

    print(
        "[pipeline] model loaded. Starting evaluation..."
    )

    max_new_tokens = model_config.get(
        "max_new_tokens",
        100
    )

    results = []

    # ======================================================
    # SEQUENCE MODE
    # ======================================================

    if mode == "sequence":

        for i, record in enumerate(records):

            messages = _build_messages(
                record["text"],
                prompt
            )

            raw = module.generate_one(
                model,
                tokenizer,
                messages,
                max_new_tokens
            )

            predicted_label = normalize_label(raw)

            reason = _extract_reason(raw)

            result = {
                "id": record.get(
                    "id",
                    str(i)
                ),
                "text": record["text"],
                "gold_label": record.get("label"),
                "language": record.get("language"),
                "raw_model_output": raw,
                "predicted_label": predicted_label,
                "reason": reason,
            }

            results.append(result)

            # --------------------------------------------------
            # SHOW RESULT
            # --------------------------------------------------

            print(
                f"\nID: {result['id']}"
            )

            print(
                f"Predicted label: "
                f"{result['predicted_label']}"
            )

            print(
                f"Reason: "
                f"{result['reason']}"
            )

            if (i + 1) % 20 == 0:

                print(
                    f"[pipeline] processed "
                    f"{i + 1}/{len(records)}"
                )

    # ======================================================
    # BATCH MODE
    # ======================================================

    elif mode == "batch":

        for start in range(
            0,
            len(records),
            batch_size
        ):

            chunk = records[
                start:start + batch_size
            ]

            messages_list = [
                _build_messages(
                    r["text"],
                    prompt
                )
                for r in chunk
            ]

            # Same model/tokenizer reused
            raw_list = module.generate_batch(
                model,
                tokenizer,
                messages_list,
                max_new_tokens
            )

            for index, (record, raw) in enumerate(
                zip(chunk, raw_list)
            ):

                predicted_label = normalize_label(raw)

                reason = _extract_reason(raw)

                result = {
                    "id": record.get(
                        "id",
                        str(start + index)
                    ),
                    "text": record["text"],
                    "gold_label": record.get("label"),
                    "language": record.get("language"),
                    "raw_model_output": raw,
                    "predicted_label": predicted_label,
                    "reason": reason,
                }

                results.append(result)

                # --------------------------------------------------
                # SHOW RESULT
                # --------------------------------------------------

                print(
                    f"\nID: {result['id']}"
                )

                print(
                    f"Predicted label: "
                    f"{result['predicted_label']}"
                )

                print(
                    f"Reason: "
                    f"{result['reason']}"
                )

            print(
                f"[pipeline] processed "
                f"{min(start + batch_size, len(records))}/"
                f"{len(records)}"
            )

    else:

        raise ValueError(
            "mode must be 'sequence' or 'batch'"
        )

    # ======================================================
    # SAVE CSV
    # ======================================================

    if output_csv is None:

        safe_name = (
            model_path
            .strip("/")
            .replace("/", "_")
        )

        output_csv = (
            f"outputs/"
            f"{safe_name}_predictions.csv"
        )

    _save_csv(
        results,
        output_csv
    )

    # ======================================================
    # METRICS
    # ======================================================

    metrics = compute_metrics(
        results
    )

    print(
        "\n===== METRICS ====="
    )

    if metrics:

        print(
            f"Accuracy: {metrics['accuracy']}"
        )

        # Your current utils.py stores these inside per_class.
        if "per_class" in metrics:

            labels = list(
                metrics["per_class"].keys()
            )

            precision = sum(
                metrics["per_class"][label]["precision"]
                for label in labels
            ) / len(labels)

            recall = sum(
                metrics["per_class"][label]["recall"]
                for label in labels
            ) / len(labels)

            f1 = sum(
                metrics["per_class"][label]["f1"]
                for label in labels
            ) / len(labels)

            print(
                f"Precision: {round(precision, 4)}"
            )

            print(
                f"Recall: {round(recall, 4)}"
            )

            print(
                f"F1: {round(f1, 4)}"
            )

    else:

        print(
            "No gold labels were provided — "
            "metrics not computed."
        )

    return results, metrics
