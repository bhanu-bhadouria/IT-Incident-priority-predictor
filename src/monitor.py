import json
import os
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "logs" / "predictions.jsonl"

def log_prediction(input_data: dict, label: str, probability: float):
    os.makedirs(LOG_PATH.parent, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        **input_data,
        "label": label,
        "probability": probability
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")