import json
import numpy as np
import pandas as pd
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "logs" / "predictions.jsonl"
CATEGORICAL_FEATURES = ["contact_type", "category", "subcategory"]
NUMERICAL_FEATURES = ["impact", "urgency", "reassignment_count", "reopen_count", "sys_mod_count"]

# Training data baseline distributions (from your actual training data)
BASELINE = {
    "contact_type": {"Phone": 0.5, "Email": 0.3, "Self-service": 0.2},
    "category": {"Network": 0.4, "Software": 0.35, "Hardware": 0.25},
    "subcategory": {"Connectivity": 0.3, "Installation": 0.3, "Failure": 0.2, "Other": 0.2},
}

def load_recent_logs(n=100):
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()[-n:]
    records = [json.loads(line) for line in lines]
    return pd.DataFrame(records)

def compute_psi(expected: dict, actual: dict) -> float:
    psi = 0.0
    all_categories = set(expected.keys()).union(set(actual.keys()))
    for category in all_categories:
        expected_pct = expected.get(category, 0.0001)
        actual_pct = actual.get(category, 0.0001)
        psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return psi

def check_drift():
    if not LOG_PATH.exists():
        print("No prediction logs found.")
        return
    recent_logs = load_recent_logs()
    for feature in CATEGORICAL_FEATURES:
        actual_distribution = recent_logs[feature].value_counts(normalize=True).to_dict()
        baseline_distribution = BASELINE[feature]
        psi_score = compute_psi(baseline_distribution, actual_distribution)
        print(f"PSI for {feature}: {psi_score:.4f}")
        if psi_score > 0.2:
            print(f"Warning: Potential drift detected in feature '{feature}' (PSI > 0.2)")