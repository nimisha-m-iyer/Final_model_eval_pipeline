"""
======================================================================
 MINIMAL LLM EVALUATION PIPELINE

 Input:  a list of dicts, e.g. [{"id": "1", "text": "...", "label": "safe"}]
 Output: a list of dicts with predictions, saved as CSV, metrics printed

 The model is loaded exactly ONCE inside evaluate() and reused for every
 record / every batch -- it is never reloaded mid-run.
======================================================================
"""

import csv
import os

from models import gemma, qwen, aya, llama
from utils import normalize_label, compute_metrics

# Add a new model by adding one line here, pointing at its own file in models/
MODEL_MODULES = {
    "gemma": gemma,
    "qwen": qwen,
    "aya": aya,
    "llama": llama,
}

DEFAULT_SYSTEM_PROMPT = "You are an expert multilingual profanity detection system."

DEFAULT_USER_TEMPLATE = (
    "Classify the following text as 'safe' or 'not safe'. "
    "Reply with ONLY the label.\n\nText: {text}"
)


def _pick_module(model_path, model_type=None):
    """
    Decides which models/<x>.py file handles this model.
    """

    if model_type:
        module = MODEL_MODULES.get(model_type.lower())

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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]


def _save_csv(results, output_csv):
    fieldnames = [
        "id",
        "text",
        "gold_label",
        "raw_model_output",
        "predicted_label"
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
                k: r.get(k)
                for k in fieldnames
            })

    print(
        f"[pipeline] predictions saved -> {output_csv}"
    )


def evaluate(
    records,
    model_config,
    prompt_config=None,
    mode="sequence",
    batch_size=8,
    output_csv=None
):
    """
    records      : list of dicts. Each needs "text". "id" and "label" optional.
    model_config : {
        "model_path": "<HF hub id or local folder path>",
        "model_type": "gemma" | "qwen" | "aya" | "llama",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "max_new_tokens": 10,
    }
    prompt_config: {"system_prompt": "...", "user_template": "...{text}..."}
    mode         : "sequence" or "batch"
    batch_size   : only used when mode == "batch"
    output_csv   : where to save predictions
    """

    prompt_config = prompt_config or {}

    model_path = model_config["model_path"]

    module = _pick_module(
        model_path,
        model_config.get("model_type")
    )

    # ---- MODEL LOADED HERE, EXACTLY ONCE ----

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

    # ------------------------------------------

    max_new_tokens = model_config.get(
        "max_new_tokens",
        10
    )

    results = []

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

            predicted_label = normalize_label(raw)

            result = {
                "id": record.get(
                    "id",
                    str(i)
                ),
                "text": record["text"],
                "gold_label": record.get("label"),
                "raw_model_output": raw,
                "predicted_label": predicted_label,
            }

            results.append(result)

            # ONLY NEW PART:
            print(
                f"\nID: {result['id']}"
            )
            print(
                f"Predicted label: {result['predicted_label']}"
            )
            print(
                f"Reason: {result['raw_model_output']}"
            )

            if (i + 1) % 20 == 0:
                print(
                    f"[pipeline] processed "
                    f"{i + 1}/{len(records)}"
                )

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
                    prompt_config
                )
                for r in chunk
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

                predicted_label = normalize_label(raw)

                result = {
                    "id": record.get(
                        "id",
                        str(start + index)
                    ),
                    "text": record["text"],
                    "gold_label": record.get("label"),
                    "raw_model_output": raw,
                    "predicted_label": predicted_label,
                }

                results.append(result)

                # ONLY NEW PART:
                print(
                    f"\nID: {result['id']}"
                )
                print(
                    f"Predicted label: {result['predicted_label']}"
                )
                print(
                    f"Reason: {result['raw_model_output']}"
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

    metrics = compute_metrics(results)

        metrics = compute_metrics(results)

    print("\n===== METRICS =====")

    if metrics:
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1: {metrics['f1']}")
    else:
        print(
            "No gold labels were provided — "
            "metrics not computed."
        )

    return results, metrics
