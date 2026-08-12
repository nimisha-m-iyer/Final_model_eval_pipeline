README.md



# LLM Model Evaluation Pipeline

A modular pipeline for evaluating different LLMs using a common `evaluate()` function.

## Repository Structure


Final_model_eval_pipeline/
│
├── models/
│   ├── gemma.py
│   ├── qwen.py
│   ├── aya.py
│   └── llama.py
│
├── pipeline.py
├── config.json
├── utils.py
├── requirements.txt
└── README.md
Description
pipeline.py
Main evaluation pipeline. It loads the configuration, loads the selected model, sends the input records to the model, and returns the model responses.

config.json
Configuration file where the evaluation parameters can be selected, including:

model
model path
torch dtype
device map
maximum number of generated tokens
prompt
evaluation mode
batch size

The configuration can be changed without modifying the pipeline code.

models/
Contains model-specific implementations. Each model file handles the model-specific loading and generation requirements, such as chat templates.
utils.py
Contains utility functions used by the pipeline.
requirements.txt
Contains the Python packages required to run the pipeline.
Running the Pipeline
1. Clone the repository
git clone https://github.com/student-nimisha/Final_model_eval_pipeline.git
cd Final_model_eval_pipeline
2. Install the requirements
pip install -r requirements.txt
3. Configure the evaluation

Edit:

config.json

Select the required model, model path, prompt, mode, batch size, and other parameters in the configuration file.

Using evaluate()

Import the evaluation function from pipeline.py:

from pipeline import evaluate

Prepare the input records:

records = [
    {
        "id": "1",
        "text": "ith oru mosham sthalam aan"
    },
    {
        "id": "2",
        "text": "Good morning!"
    }
]

Run evaluation:

results = evaluate(
    records=records
)

The pipeline reads the model and evaluation settings from config.json.

The returned results contains the model's raw response for each input along with its corresponding ID.

Example:

[
    {
        "id": "1",
        "response": "..."
    },
    {
        "id": "2",
        "response": "..."
    }
]

No changes to pipeline.py are required when changing the evaluation configuration. Edit config.json instead.
