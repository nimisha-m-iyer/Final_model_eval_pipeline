**LLM Model Evaluation Pipeline**

A modular pipeline for evaluating LLMs using a common evaluate() function.

**Usage**
1. Clone the repository
git clone https://github.com/student-nimisha/Final_model_eval_pipeline.git

cd Final_model_eval_pipeline

3. Install dependencies
pip install -r requirements.txt

4. Configure the evaluation

Edit config.json to select the required:

Model and model path
Prompt
Mode (batch / sequence)
Batch size
Generation parameters
4. Run evaluation
from pipeline import evaluate

records = [
    {"id": "1", "text": "ith oru mosham sthalam aan"},
    {"id": "2", "text": "Good morning!"}
]

**EVALUATE FUNCTION:**

def evaluate(
    records,
    config_path="config.json"
):

    config = load_config(config_path)

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

    if model_type not in MODEL_MODULES:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: {list(MODEL_MODULES.keys())}"
        )

    module = MODEL_MODULES[model_type]

    print(f"[pipeline] model type: {model_type}")
    print(f"[pipeline] model path: {model_path}")
    print("[pipeline] loading model...")

    model, tokenizer = module.load(
        model_path,
        torch_dtype,
        device_map
    )

    print("[pipeline] model loaded.")

    results = []

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

            results.append(result)

            print(f"\nID: {result['id']}")
            print(f"Response: {result['response']}")

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

                results.append(result)

                print(f"\nID: {result['id']}")
                print(f"Response: {result['response']}")

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



results = evaluate(records)

The pipeline reads all configuration from config.json and returns the model's raw responses:

[
    {"id": "1", "response": "..."},
    {"id": "2", "response": "..."}
]e evaluation configuration. Edit config.json instead.
