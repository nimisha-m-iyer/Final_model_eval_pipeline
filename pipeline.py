"""
Minimal LLM Evaluation Pipeline

The user explicitly provides:

    model_type
    model_path
    prompt
    mode
    batch_size

Example:

    model_config = {
        "model_type": "gemma",
        "model_path": "google/gemma-3-4b-it",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "max_new_tokens": 100
    }

The model is loaded exactly once.

The same model and tokenizer are reused
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
# MODEL FILES
# ============================================================
#
# model_type = "gemma"
#        ↓
# models/gemma.py
#
# model_type = "qwen"
#        ↓
# models/qwen.py
#
# model_type = "aya"
#        ↓
# models/aya.py
#
# model_type = "llama"
#        ↓
# models/llama.py
#
# This is NOT the model weights.
# It only selects the Python file containing
# model-specific code.
# ============================================================

MODEL_MODULES = {
    "gemma": gemma,
    "qwen": qwen,
    "aya": aya,
    "llama": llama,
}


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

        Example:

        [
            {
                "id": "1",
                "text": "നീ ഒരു മൈരൻ ആണ്",
                "label": "not safe",
                "language": "ml"
            }
        ]


    model_config:

        {
            "model_type": "gemma",

            "model_path": "google/gemma-3-4b-it",

            "torch_dtype": "bfloat16",

            "device_map": "auto",

            "max_new_tokens": 100
        }


    prompt:
        The complete prompt supplied by the user.

        {text} is replaced with record["text"].


    mode:
        "sequence" or "batch"


    batch_size:
        Used only for batch mode.


    output_csv:
        Optional CSV path.
    """

    # ========================================================
    # 1. GET MODEL INFORMATION
    # ========================================================

    model_type = model_config["model_type"]

    model_path = model_config["model_path"]

    max_new_tokens = model_config.get(
        "max_new_tokens",
        100
    )

    # ========================================================
    # 2. SELECT MODEL-SPECIFIC PYTHON FILE
    # ========================================================
    #
    # Example:
    #
    # model_type = "gemma"
    #
    # module = models/gemma.py
    #
    # Nothing is inferred from model_path.
    # ========================================================

    if model_type not in MODEL_MODULES:

        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: "
            f"{list(MODEL_MODULES.keys())}"
        )

    module = MODEL_MODULES[
        model_type
    ]

    # ========================================================
    # 3. LOAD THE MODEL
    # ========================================================
    #
    # model_path is passed DIRECTLY to the
    # model-specific load() function.
    #
    # For example:
    #
    # "google/gemma-3-4b-it"
    #
    # goes directly to:
    #
    # models/gemma.py
    #
    # and then:
    #
    # AutoTokenizer.from_pretrained(model_path)
    #
    # AutoModelForCausalLM.from_pretrained(model_path)
    #
    # Hugging Face downloads the weights if they
    # are not already cached.
    #
    # This happens ONLY ONCE.
    # ========================================================

    print(
        f"[pipeline] model type: {model_type}"
    )

    print(
        f"[pipeline] model path: {model_path}"
    )

    print(
        "[pipeline] loading model..."
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
        "[pipeline] model loaded. "
        "Starting evaluation..."
    )

    # ========================================================
    # 4. RESULTS
    # ========================================================

    results = []

    # ========================================================
    # 5. SEQUENCE MODE
    # ========================================================

    if mode == "sequence":

        for i, record in enumerate(records):

            # ------------------------------------------------
            # Insert text into user's prompt
            # ------------------------------------------------

            current_prompt = prompt.format(
                text=record["text"]
            )

            # ------------------------------------------------
            # Send prompt to model-specific implementation
            # ------------------------------------------------

            raw = module.generate_one(
                model,
                tokenizer,
                current_prompt,
                max_new_tokens
            )

            # ------------------------------------------------
            # Extract label and reason
            # ------------------------------------------------

            predicted_label = normalize_label(
                raw
            )

            reason = extract_reason(
                raw
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

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

                "predicted_label":
                    predicted_label,

                "reason":
                    reason
            }

            results.append(result)

            # ------------------------------------------------
            # Show result in editor
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
    # 6. BATCH MODE
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

            # ------------------------------------------------
            # Create prompts for this batch
            # ------------------------------------------------

            prompts = [
                prompt.format(
                    text=record["text"]
                )
                for record in chunk
            ]

            # ------------------------------------------------
            # Send entire batch to model
            # ------------------------------------------------

            raw_list = module.generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens
            )

            # ------------------------------------------------
            # Process results
            # ------------------------------------------------

            for index, (record, raw) in enumerate(
                zip(
                    chunk,
                    raw_list
                )
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

                    "predicted_label":
                        predicted_label,

                    "reason":
                        reason
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

    # ========================================================
    # 7. SAVE CSV
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
    # 8. METRICS
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


def _save_csv(
    results,
    output_csv
):

    """
    Save only the requested output columns.
    """

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
                "predicted_label":
                    result.get("predicted_label"),
                "reason":
                    result.get("reason"),
                "language":
                    result.get("language")
            })

    print(
        f"[pipeline] predictions saved -> "
        f"{output_csv}"
    )
