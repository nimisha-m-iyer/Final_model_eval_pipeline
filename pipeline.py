"""
MINIMAL LLM EVALUATION PIPELINE
"""

import csv
import os

from models import gemma, qwen, aya, llama
from utils import parse_model_output, compute_metrics


# ---------------------------------------------------------
# MODEL MODULES
# ---------------------------------------------------------

MODEL_MODULES = {
    "gemma": gemma,
    "qwen": qwen,
    "aya": aya,
    "llama": llama,
}


# ---------------------------------------------------------
# DEFAULT PROMPT
# ---------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert multilingual profanity detection system."
)

DEFAULT_USER_TEMPLATE = """
Classify the following text as either "safe" or "not safe".

Return ONLY valid JSON in exactly this format:

{
    "label": "safe",
    "reason": "short explanation"
}

The label must be exactly one of:
- safe
- not safe

Text:
{text}
"""


# ---------------------------------------------------------
# SELECT MODEL
# ---------------------------------------------------------

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
        f"Pass model_config['model_type'] explicitly."
    )


# ---------------------------------------------------------
# BUILD PROMPT
# ---------------------------------------------------------

def _build_messages(text, prompt_config):

    system_prompt = prompt_config.get(
        "system_prompt",
        DEFAULT_SYSTEM_PROMPT
    )

    user_template = prompt_config.get(
        "user_template",
        DEFAULT_USER_TEMPLATE
    )

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_template.format(
                text=text
            )
        }
    ]


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

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
                "id": result.get("id", ""),
                "predicted_label": result.get(
                    "predicted_label",
                    ""
                ),
                "reason": result.get(
                    "reason",
                    ""
                ),
                "language": result.get(
                    "language",
                    ""
                )
            })

    print(
        f"\n[pipeline] predictions saved -> {output_csv}"
    )


# ---------------------------------------------------------
# PRINT PREDICTIONS
# ---------------------------------------------------------

def _print_prediction(result):

    print(
        f"\nID        : {result['id']}"
    )

    print(
        f"Language  : {result['language']}"
    )

    print(
        f"Prediction: {result['predicted_label']}"
    )

    print(
        f"Reason    : {result['reason']}"
    )

    print("-" * 60)


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def evaluate(
    records,
    model_config,
    prompt_config=None,
    mode="sequence",
    batch_size=8,
    output_csv=None
):

    prompt_config = prompt_config or {}

    model_path = model_config["model_path"]

    module = _pick_module(
        model_path,
        model_config.get("model_type")
    )

    # -----------------------------------------------------
    # LOAD MODEL ONCE
    # -----------------------------------------------------

    print(
        f"\n[pipeline] loading model from: {model_path}"
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

    max_new_tokens = model_config.get(
        "max_new_tokens",
        50
    )

    results = []

    # -----------------------------------------------------
    # SEQUENCE
    # -----------------------------------------------------

    if mode == "sequence":

        for i, record in enumerate(records):

            messages = _build_messages(
                record["text"],
                prompt_config
            )

            raw = module.generate_one(
                model,
                tokenizer,
                messages,
                max_new_tokens
            )

            predicted_label, reason = (
                parse_model_output(raw)
            )

            result = {
                "id": record.get(
                    "id",
                    str(i)
                ),
                "text": record["text"],
                "gold_label": record.get("label"),
                "predicted_label": predicted_label,
                "reason": reason,
                "language": record.get(
                    "language",
                    ""
                )
            }

            results.append(result)

            # SHOW RESULT IN EDITOR
            _print_prediction(result)

    # -----------------------------------------------------
    # BATCH
    # -----------------------------------------------------

    elif mode == "batch":

        for start in range(
            0,
            len(records),
            batch_size
        ):

            chunk = records[
                start:start + batch_size
            ]

            print(
                f"\n[pipeline] "
                f"processing batch "
                f"{start // batch_size + 1}"
            )

            messages_list = [
                _build_messages(
                    record["text"],
                    prompt_config
                )
                for record in chunk
            ]

            raw_list = module.generate_batch(
                model,
                tokenizer,
                messages_list,
                max_new_tokens
            )

            for index, (record, raw) in enumerate(
                zip(chunk, raw_list)
            ):

                predicted_label, reason = (
                    parse_model_output(raw)
                )

                result = {
                    "id": record.get(
                        "id",
                        str(start + index)
                    ),
                    "text": record["text"],
                    "gold_label": record.get("label"),
                    "predicted_label": predicted_label,
                    "reason": reason,
                    "language": record.get(
                        "language",
                        ""
                    )
                }

                results.append(result)

                # SHOW RESULT IN EDITOR
                _print_prediction(result)

            print(
                f"[pipeline] processed "
                f"{min(start + batch_size, len(records))}/"
                f"{len(records)}"
            )

    else:

        raise ValueError(
            "mode must be 'sequence' or 'batch'"
        )

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    metrics = compute_metrics(results)

    print("\n")
    print("=" * 60)
    print("FINAL METRICS")
    print("=" * 60)

    if metrics:

        print(
            f"Accuracy  : {metrics['accuracy']}"
        )

        print(
            f"Precision : {metrics['precision']}"
        )

        print(
            f"Recall    : {metrics['recall']}"
        )

        print(
            f"F1        : {metrics['f1']}"
        )

    else:

        print(
            "No gold labels were provided."
        )

    print("=" * 60)

    return results, metrics
