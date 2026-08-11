"""
Minimal LLM Evaluation Pipeline

Input:
    A list of dictionaries.

Example:
    records = [
        {
            "id": "1",
            "text": "some text",
            "label": "safe",
            "language": "ml"
        }
    ]

The model is loaded exactly once and reused
for all records and batches.
"""

import csv
import importlib
import os

from utils import (
    normalize_label,
    extract_reason,
    compute_metrics
)


def _load_model_module(model_type):
    """
    Loads the model-specific file.

    Example:
        "gemma" -> models/gemma.py
        "qwen"  -> models/qwen.py
        "aya"   -> models/aya.py
        "llama" -> models/llama.py
    """

    return importlib.import_module(
        f"models.{model_type.lower()}"
    )


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

        for result in results:

            writer.writerow({
                "id": result.get("id"),
                "predicted_label": result.get(
                    "predicted_label"
                ),
                "reason": result.get("reason"),
                "language": result.get("language")
            })

    print(
        f"[pipeline] predictions saved -> {output_csv}"
    )


def evaluate(
    records,
    model_config,
    prompt,
    mode="sequence",
    batch_size=8,
    output_csv=None
):
    """
    records:
        List of dictionaries.

        Required:
            text

        Optional:
            id
            label
            language


    model_config:

        {
            "model_path": "google/gemma-3-4b-it",
            "model_type": "gemma",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 100
        }


    prompt:
        One complete prompt supplied by the user.

        {text} is replaced with record["text"].


    mode:
        "sequence" or "batch"


    batch_size:
        Used only for batch mode.


    output_csv:
        Optional path for the predictions CSV.
    """

    # ==================================================
    # MODEL CONFIG
    # ==================================================

    model_path = model_config["model_path"]

    model_type = model_config["model_type"]

    max_new_tokens = model_config.get(
        "max_new_tokens",
        100
    )

    # ==================================================
    # LOAD MODEL-SPECIFIC MODULE
    # ==================================================

    module = _load_model_module(
        model_type
    )

    # ==================================================
    # LOAD MODEL ONCE
    # ==================================================

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
        )
    )

    print(
        "[pipeline] model loaded. Starting evaluation..."
    )

    results = []

    # ==================================================
    # SEQUENCE MODE
    # ==================================================

    if mode == "sequence":

        for i, record in enumerate(records):

            current_prompt = prompt.format(
                text=record["text"]
            )

            raw = module.generate_one(
                model,
                tokenizer,
                current_prompt,
                max_new_tokens
            )

            predicted_label = normalize_label(
                raw
            )

            reason = extract_reason(
                raw
            )

            result = {
                "id": record.get(
                    "id",
                    str(i)
                ),
                "text": record["text"],
                "gold_label": record.get(
                    "label"
                ),
                "language": record.get(
                    "language"
                ),
                "raw_model_output": raw,
                "predicted_label": predicted_label,
                "reason": reason
            }

            results.append(result)

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

    # ==================================================
    # BATCH MODE
    # ==================================================

    elif mode == "batch":

        for start in range(
            0,
            len(records),
            batch_size
        ):

            chunk = records[
                start:start + batch_size
            ]

            prompts = [
                prompt.format(
                    text=record["text"]
                )
                for record in chunk
            ]

            raw_list = module.generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens
            )

            for index, (record, raw) in enumerate(
                zip(chunk, raw_list)
            ):

                predicted_label = normalize_label(
                    raw
                )

                reason = extract_reason(
                    raw
                )

                result = {
                    "id": record.get(
                        "id",
                        str(start + index)
                    ),
                    "text": record["text"],
                    "gold_label": record.get(
                        "label"
                    ),
                    "language": record.get(
                        "language"
                    ),
                    "raw_model_output": raw,
                    "predicted_label": predicted_label,
                    "reason": reason
                }

                results.append(result)

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

    # ==================================================
    # SAVE CSV
    # ==================================================

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

    # ==================================================
    # METRICS
    # ==================================================

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

        print(
            f"Precision: {metrics['precision']}"
        )

        print(
            f"Recall: {metrics['recall']}"
        )

        print(
            f"F1: {metrics['f1']}"
        )

    else:

        print(
            "No gold labels were provided — "
            "metrics not computed."
        )

    return results, metrics
