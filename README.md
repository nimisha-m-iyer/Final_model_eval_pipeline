Usage
1. Clone the repository
git clone https://github.com/student-nimisha/Final_model_eval_pipeline.git
cd Final_model_eval_pipeline
2. Install dependencies
pip install -r requirements.txt
3. Configure the evaluation

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

results = evaluate(records)

The pipeline reads all configuration from config.json and returns the model's raw responses:

[
    {"id": "1", "response": "..."},
    {"id": "2", "response": "..."}
]e evaluation configuration. Edit config.json instead.
