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

Output:
    - predicted label
    - reason
    - CSV
    - accuracy
    - precision
    - recall
    - F1

The model is loaded exactly once and reused
for all records and batches.
"""

import csv
import os

from models import gemma, qwen, aya, llama

from utils import (
    normalize_label,
    extract_reason,
    compute_metrics
)


# ============================================================
# MODEL MODULES
# ============================================================
#
# This maps the model name/type to its Python file.
#
# "gemma" -> models/gemma.py
# "qwen"  -> models/qwen.py
# "aya"   -> models/aya.py
# "llama" -> models/llama.py
#
# It does NOT load the model weights.
# The model weights are loaded later using model_path.
# ============================================================

MODEL_MODULES = {
    "gemma": gemma,
    "qwen": qwen,
    "aya": aya,
    "llama": llama,
}


def _pick_module(model_path, model_type=None):
    """
    Decide which model-specific Python file to use.

    If model_type is given:
        use it directly.

    Otherwise:
        try to detect the model type from model_path.

    Example:

        model_path = "google/gemma-3-4b-it"

        -> "gemma" found in path
        -> models/gemma.py

    """

    # --------------------------------------------------------
    # Explicit model type
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Try to detect model type from model path
    # --------------------------------------------------------

    name = model_path.lower()

    for key, module in MODEL_MODULES.items():

        if key in name:

            return module

    # --------------------------------------------------------
    # Could not determine model
    # --------------------------------------------------------

    raise ValueError(
        f"Could not detect model type from '{model_path}'. "
        f"Either include one of "
        f"{list(MODEL_MODULES.keys())} "
        f"in the path or provide "
        f"model_config['model_type'] explicitly."
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

        {text} is replaced by record["text"].


    mode:
        "sequence" or "batch"


    batch_size:
        Used only for batch mode.


    output_csv:
        Optional path for the predictions CSV.
    """

    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    model_path = model_config["model_path"]

    model_type = model_config.get(
        "model_type"
    )

    max_new_tokens = model_config.get(
        "max_new_tokens",
        100
    )

    # ========================================================
    # SELECT MODEL-SPECIFIC MODULE
    # ========================================================

    module = _pick_module(
        model_path,
        model_type
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================
    #
    # THIS HAPPENS ONLY ONCE.
    #
    # The same model and tokenizer are reused
    # for every sequence or every batch.
    # ========================================================

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

    # ========================================================
    # SEQUENCE MODE
    # ========================================================

    if mode == "sequence":

        for i, record in enumerate(records):

            # Insert the current text into the
            # user-provided prompt.

            current_prompt = prompt.format(
                text=record["text"]
            )

            # Model-specific file handles:
            #
            # - chat template
            # - tokenization
            # - generation

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

            # ------------------------------------------------
            # SHOW RESULT IN NOTEBOOK
            # ------------------------------------------------

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

    # ========================================================
    # BATCH MODE
    # ========================================================

    elif mode == "batch":

        for start in range(
            0,
            len(records),
            batch_size
        ):

            chunk = records[
                start:start + batch_size
            ]

            # Build one prompt per record.

            prompts = [
                prompt.format(
                    text=record["text"]
                )
                for record in chunk
            ]

            # Model-specific batch generation.

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

                # ------------------------------------------------
                # SHOW RESULT IN NOTEBOOK
                # ------------------------------------------------

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

    # ========================================================
    # SAVE CSV
    # ========================================================

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

    # ========================================================
    # METRICS
    # ========================================================

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
