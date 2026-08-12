"""
Minimal LLM Evaluation Pipeline

Input:
    [
        {"id": "1", "text": "Hello"},
        {"id": "2", "text": "Some text"}
    ]

Output:
    [
        {"id": "1", "response": "..."},
        {"id": "2", "response": "..."}
    ]

The model is loaded exactly once per evaluate() call.
"""

from models import gemma, qwen, aya, llama


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
# EVALUATION
# ========================================================

def evaluate(
    records,
    model_config,
    prompt,
    mode="sequence",
    batch_size=8
):

    """
    records:
        List of dictionaries.

        Example:
        [
            {"id": "1", "text": "Hello"},
            {"id": "2", "text": "Some text"}
        ]

    model_config:
        Example:
        {
            "model_type": "gemma",
            "model_path": "google/gemma-3-4b-it",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 100
        }

    prompt:
        Prompt string containing {text}.

        Example:
        "Classify this text:\n\nText: {text}"

    mode:
        "sequence" or "batch"

    batch_size:
        Used only when mode="batch".

    Returns:
        [
            {
                "id": "...",
                "response": "raw model output"
            }
        ]
    """

    # ====================================================
    # 1. READ MODEL CONFIG
    # ====================================================

    model_type = model_config["model_type"]
    model_path = model_config["model_path"]

    module = MODEL_MODULES[model_type]


    # ====================================================
    # 2. MODEL SETTINGS
    # ====================================================

    torch_dtype = model_config.get(
        "torch_dtype",
        "bfloat16"
    )

    device_map = model_config.get(
        "device_map",
        "auto"
    )

    max_new_tokens = model_config.get(
        "max_new_tokens",
        100
    )


    # ====================================================
    # 3. LOAD MODEL
    # ====================================================

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
        "[pipeline] model loaded. "
        "Starting evaluation..."
    )


    # ====================================================
    # 4. STORE RESULTS
    # ====================================================

    results = []


    # ====================================================
    # 5. SEQUENCE MODE
    # ====================================================

    if mode == "sequence":

        for i, record in enumerate(records):

            current_prompt = prompt.format(
                text=record["text"]
            )

            raw_response = module.generate_one(
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

                "response": raw_response
            }

            results.append(result)

            print()
            print(
                f"ID: {result['id']}"
            )
            print(
                f"Response: {result['response']}"
            )

        print()
        print(
            f"[pipeline] processed "
            f"{len(results)}/{len(records)}"
        )


    # ====================================================
    # 6. BATCH MODE
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
                raw_response
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

                    "response": raw_response
                }

                results.append(result)

                print()
                print(
                    f"ID: {result['id']}"
                )
                print(
                    f"Response: {result['response']}"
                )

            print(
                f"[pipeline] processed "
                f"{min(start + batch_size, len(records))}/"
                f"{len(records)}"
            )


    # ====================================================
    # 7. INVALID MODE
    # ====================================================

    else:

        raise ValueError(
            "mode must be 'sequence' or 'batch'"
        )


    # ====================================================
    # 8. RETURN RAW RESULTS
    # ====================================================

    return results
