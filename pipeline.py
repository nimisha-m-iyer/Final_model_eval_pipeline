"""
Minimal LLM evaluation pipeline.

Configuration is loaded from config.json when this
module is imported.

Input:
    list of dictionaries containing:
        id
        text
        type (optional)

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
# LOAD CONFIG ONCE
# ========================================================

def load_config(config_path="config.json"):

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


CONFIG = load_config()


# ========================================================
# MODEL SETTINGS
# ========================================================

model_type = CONFIG["model_type"]

model_path = CONFIG["model_path"]

torch_dtype = CONFIG.get(
    "torch_dtype",
    "bfloat16"
)

device_map = CONFIG.get(
    "device_map",
    "auto"
)


# ========================================================
# SELECT MODEL MODULE
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
# LOAD MODEL ONCE
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
    torch_dtype,
    device_map
)

print(
    "[pipeline] model loaded."
)


# ========================================================
# EVALUATE
# ========================================================

def evaluate(records):

    # ----------------------------------------------------
    # Get settings from config
    # ----------------------------------------------------

    max_new_tokens = CONFIG.get(
        "max_new_tokens",
        100
    )

    prompt = CONFIG["prompt"]

    batch_size = CONFIG.get(
        "batch_size",
        1
    )

    # ----------------------------------------------------
    # Generate responses
    # ----------------------------------------------------

    results = []

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
                text=record["text"],
                type=record.get(
                    "type",
                    ""
                )
            )
            for record in chunk
        ]

        responses = module.generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens
        )

        # ------------------------------------------------
        # Store results
        # ------------------------------------------------

        for index, (
            record,
            response
        ) in enumerate(
            zip(
                chunk,
                responses
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

    return results
