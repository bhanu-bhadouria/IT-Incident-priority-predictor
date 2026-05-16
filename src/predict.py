import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"


def predict_priority(
    impact,
    urgency,
    reassignment_count,
    reopen_count,
    contact_type,
    category,
    subcategory,
    opened_at,
    sys_mod_count,
    notify,
    location = 'Unknown',
    assignment_group = 'Unassigned',
):
    # --- Build raw 1-row DataFrame ---
    df = pd.DataFrame([{
        "impact":             impact,
        "urgency":            urgency,
        "reassignment_count": reassignment_count,
        "reopen_count":       reopen_count,
        "contact_type":       contact_type,
        "category":           category,
        "subcategory":        subcategory,
        "location":           location,
        "assignment_group":   assignment_group,
        "opened_at":          opened_at,
        "sys_mod_count":      sys_mod_count,
        "notify":             notify,
    }])

    # --- Temporal features derived from opened_at ---
    dt         = pd.to_datetime(df["opened_at"])
    hour       = dt.dt.hour
    day_of_week = dt.dt.dayofweek  # Monday=0, Sunday=6

    df["is_business_hours"] = ((hour >= 9) & (hour < 17) & (day_of_week < 5)).astype(int)
    df["is_weekend"]        = (day_of_week >= 5).astype(int)
    df["is_night"]          = ((hour < 8) | (hour >= 18)).astype(int)
    df["quarter"]           = dt.dt.quarter
    df["hour"]              = hour
    df["day_of_week"]       = day_of_week
    df = df.drop(columns=["opened_at"])

    # --- Engineered features ---
    df["impact_urgency_score"] = df["impact"] + df["urgency"]
    df["escalation_risk"]      = (
        (df["reassignment_count"] >= 2) & (df["reopen_count"] >= 1)
    ).astype(int)

    # --- Encode categorical columns using saved encoders ---
    encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
    for col in ["contact_type", "category", "subcategory", "location", "assignment_group"]:
        le = encoders[col]
        # map unseen values to the most frequent known class rather than crashing
        df[col] = df[col].apply(
            lambda v, enc=le: enc.transform([v])[0] if v in enc.classes_ else 0
        )

    # --- Reorder columns to match training feature order ---
    feature_list = joblib.load(MODEL_DIR / "feature_list.joblib")
    df = df[feature_list]

    # --- Load model + threshold → predict ---
    model     = joblib.load(MODEL_DIR / "best_model_final.joblib")
    threshold = joblib.load(MODEL_DIR / "threshold.joblib")

    raw_probability = model.predict_proba(df)[0, 1]
    predicted_label = "High Priority" if raw_probability >= threshold else "Normal"
    confidence_pct  = round(raw_probability * 100, 2)

    return predicted_label, confidence_pct, raw_probability, df

if __name__ == "__main__":
    label, confidence, prob, _ = predict_priority(
        impact=3,
        urgency=2,
        reassignment_count=3,
        reopen_count=2,
        contact_type="Phone",
        category="Network",
        subcategory="Connectivity",
        opened_at="2024-01-15 02:30:00",   # 2am, Monday
        sys_mod_count=5,
        notify=1,
    )
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence}%")
    print(f"Probability: {prob:.4f}")
