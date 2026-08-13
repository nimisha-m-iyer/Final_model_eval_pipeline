**LLM Model Evaluation Pipeline**

A modular pipeline for evaluating LLMs using a common evaluate() function.

**Usage**
1. Clone the repository
git clone https://github.com/student-nimisha/Final_model_eval_pipeline.git

2. cd Final_model_eval_pipeline

3. Install dependencies
pip install -r requirements.txt

4.Edit config.json to select the required:

    Model and model path
    Prompt
    Batch size
    Generation parameters

5. From pipeline import evaluate


6.Pass the inputs as "records" in dictionary format

example:
records = [
    {"id": "1", "text": "ith oru mosham sthalam aan"},
    {"id": "2", "text": "Good morning!"}
]





**EVALUATE FUNCTION:**


def evaluate(records):

    # ----------------------------------------------------
    # Get settings from config
    # ----------------------------------------------------

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
    # Select model-specific module
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
    # Load model once
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
                text=record["text"]
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



7.results = evaluate(records)

The pipeline reads all configuration from config.json and returns the model's raw responses:

[
    {"id": "1", "response": "..."},
    {"id": "2", "response": "..."}
]
