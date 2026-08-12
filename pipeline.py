"""
Minimal LLM evaluation pipeline.

The configuration is read from config.json.

Input:
    list of dictionaries containing:
        id
        text

Output:
    list of dictionaries containing:
        id
        response
"""

import json

from models import (
    gemma,
    qwen,
    aya,
    llama
)


# ========================================================
# MODEL MODULES
# ========================================================

MODEL_MODULES = {
    "gemma": gemma,
    "qwen": qwen,
    "aya": aya,
    "llama": llama,
}


# ========================================================
# LOAD CONFIG
# ========================================================

def load_config(config_path="config.json"):

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ========================================================
# EVALUATE
# ========================================================

def evaluate(
    records,
    config_path="config.json"
):

    # ----------------------------------------------------
    # Read configuration
    # ----------------------------------------------------

    config = load_config(
        config_path
    )

    model_type = config["model_type"]
    model_path = config["model_path"]

    torch_dtype = config.get(
        "torch_dtype",
        "bfloat16"
    )

    device_map = config.get(
        "device_map",
        "auto"
    )

    max_new_tokens = config.get(
        "max_new_tokens",
        100
    )

    prompt = config["prompt"]

    mode = config.get(
        "mode",
        "sequence"
    )

    batch_size = config.get(
        "batch_size",
        1
    )

    # ----------------------------------------------------
    # Select model-specific file
    # ----------------------------------------------------

    if model_type not in MODEL_MODULES:

        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: "
            f"{list(MODEL_MODULES.keys())}"
        )

    module = MODEL_MODULES[
        model_type
    ]

    # ----------------------------------------------------
    # Load model ONCE
    # ----------------------------------------------------

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
        torch_dtype,
        device_map
    )

    print(
        "[pipeline] model loaded."
    )

    # ----------------------------------------------------
    # RESULTS
    # ----------------------------------------------------

    results = []

    # ====================================================
    # SEQUENCE MODE
    # ====================================================

    if mode == "sequence":

        for i, record in enumerate(records):

            current_prompt = prompt.format(
                text=record["text"]
            )

            response = module.generate_one(
                model,
                tokenizer,
                current_prompt,
                max_new_tokens
            )

            result = {
                "id": record.get(
                    "id",
                    str(i)
                ),
                "response": response
            }

            results.append(
                result
            )

            print(
                f"\nID: {result['id']}"
            )

            print(
                f"Response: {result['response']}"
            )

    # ====================================================
    # BATCH MODE
    # ====================================================

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

            for index, (
                record,
                response
            ) in enumerate(
                zip(
                    chunk,
                    raw_list
                )
            ):

                result = {
                    "id": record.get(
                        "id",
                        str(start + index)
                    ),
                    "response": response
                }

                results.append(
                    result
                )

                print(
                    f"\nID: {result['id']}"
                )

                print(
                    f"Response: {result['response']}"
                )

            print(
                f"\n[pipeline] processed "
                f"{min(start + batch_size, len(records))}/"
                f"{len(records)}"
            )

    else:

        raise ValueError(
            "mode must be 'sequence' or 'batch'"
        )

    return results
