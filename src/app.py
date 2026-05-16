import pandas as pd
import shap
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from datetime import datetime, date, time
from predict import predict_priority
import joblib

MODEL_DIR = Path(__file__).parent.parent / "models"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "best_model_final.joblib")

FEATURE_LABELS = {
    "impact":	"Business Impact (1=High, 2=Medium, 3=Low)",
    "urgency":	"Resolution Urgency (1=High, 2=Medium, 3=Low)",
    "impact_urgency_score":	"Combined Impact + Urgency Score",
    "reassignment_count":	"Number of Times Ticket Was Reassigned",
    "reopen_count":	"Number of Times Ticket Was Reopened",
    "escalation_risk":	"Escalation Risk Flag (reassigned ≥2 times AND reopened)",
    "contact_type":	"How the Incident Was Reported (Phone, Email, Self-service, etc.)",
    "category":	"Incident Category (e.g. Network, Hardware, Software)",
    "subcategory":	"Incident Subcategory (e.g. Connectivity, Crash, Login)",
    "location":	"Site or Office Location of the Affected User",
    "assignment_group":	"Team Currently Responsible for the Ticket",
    "sys_mod_count":	"Number of Times the Ticket Record Has Been Updated",
    "notify":	"Notification Preference Set on the Ticket",
    "is_business_hours":	"Was Ticket Opened During Business Hours? (9am–5pm, Mon–Fri)",
    "is_weekend":	"Was Ticket Opened on a Weekend?",
    "is_night":	"Was Ticket Opened Overnight? (Before 8am or After 6pm)",
    "quarter":	"Calendar Quarter When Ticket Was Opened (Q1–Q4)",
    "hour":	"Hour of Day When Ticket Was Opened (0–23)",
    "day_of_week":	"Day of Week When Ticket Was Opened (0=Monday, 6=Sunday)"
}
encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
st.set_page_config(page_title="IT Incident Priority Predictor ", layout="wide")
st.title("IT Incident Priority Predictor")
st.subheader("Assess ticket priority at any stage of its lifecycle")

# ── Sidebar inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ticket Details")

    # Categoricals — st.selectbox
    category     = st.selectbox("Category", options     = list(encoders['category'].classes_))
    subcategory  = st.selectbox("Subcategory",  options  = list(encoders['subcategory'].classes_))
    contact_type = st.selectbox("Contact Type", options = list(encoders['contact_type'].classes_))

    # Severity — st.slider (1=High, 3=Low in ServiceNow)
    impact  = st.slider("Impact",  min_value=1, max_value=3, value=3)
    urgency = st.slider("Urgency", min_value=1, max_value=3, value=3)

    # Counts — st.number_input
    reassignment_count = st.number_input("Reassignment Count", min_value=0, max_value=50,  value=0, step=1)
    reopen_count       = st.number_input("Reopen Count",       min_value=0, max_value=20,  value=0, step=1)
    sys_mod_count      = st.number_input("Sys Mod Count",      min_value=0, max_value=200, value=0, step=1)
    notify             = st.number_input("Notify",             min_value=0, max_value=3,   value=0, step=1)

    # Datetime

    opened_date = st.date_input("Opened Date", value=datetime.now().date())
    opened_time = st.time_input("Opened Time", value=time(9, 0))
    opened_at   = datetime.combine(opened_date, opened_time)

tab1, tab2 = st.tabs(['Single Ticket Prediction', 'Batch Upload'])

with tab1:
    if st.sidebar.button("Predict Priority", type="primary"):
        label, conf, prob, processed_df = predict_priority(
            impact=impact, urgency=urgency,
            reassignment_count=reassignment_count,
            reopen_count=reopen_count,
            contact_type=contact_type,
            category=category,
            subcategory=subcategory,
            opened_at=str(opened_at),
            sys_mod_count=sys_mod_count,
            notify=notify,
        )

        if label == "High Priority":
            st.error("🚨 HIGH PRIORITY — Escalate Immediately")
            st.metric("Confidence", f"{conf:.2f}%")
            st.info("Recommended action: Page on-call engineer")
        else:
            st.success("✅ Normal Priority")
            st.metric("Confidence", f"{conf:.2f}%")
            st.info("Recommended action: Assign to standard queue")

        st.progress(prob.astype(float), text=f"Probability: {prob:.4f}  |  Threshold: 0.7",)
        st.caption("Model flags High Priority when probability ≥ 0.7. "
                "A higher threshold means fewer false alarms but some critical "
                "tickets may be missed.")

        st.caption("Influence Score shows how much each factor pushed the prediction. "
            "Longer bar = stronger influence on this specific ticket.")

        with st.expander("Why did the model predict this?"):
            model = load_model()
            explainer    = shap.TreeExplainer(model)
            shap_vals    = explainer.shap_values(processed_df)  # shape: (1, 18)

            shap_df = pd.DataFrame({
                "feature":    processed_df.columns,
                "shap_value": shap_vals[0],
            })
            shap_df["label"] = shap_df["feature"].map(FEATURE_LABELS)
            shap_df = shap_df.rename(columns={"shap_value": "Influence Score"})
            shap_df["Influence Score"] = shap_df["Influence Score"].round(3)
            shap_df = shap_df.sort_values("Influence Score", ascending=False)

            st.markdown("**Top factors pushing toward 🚨 High Priority:**")
            st.dataframe(shap_df[shap_df["Influence Score"] > 0].head(5)[["label","Influence Score"]],
                        hide_index=True)
            pushers = shap_df[shap_df["Influence Score"] > 0].head(5).set_index("label")
            st.bar_chart(pushers["Influence Score"])

            st.markdown("**Factors pushing toward ✅ Normal:**")
            st.dataframe(shap_df[shap_df["Influence Score"] < 0].tail(3)[["label","Influence Score"]],
                        hide_index=True)
            dampeners = shap_df[shap_df["Influence Score"] < 0].tail(3).set_index("label")
            st.bar_chart(dampeners["Influence Score"].abs())  # abs() so bars go right, not left
    else:
        st.info("Fill in the ticket details on the left and click Predict.")

with tab2:
    st.subheader("Batch Ticket Assessment")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)

        # 1. Check required columns exist
        required_cols = ['impact', 'urgency', 'reassignment_count', 'reopen_count',
                         'contact_type', 'category', 'subcategory',
                         'opened_at', 'sys_mod_count', 'notify']
        missing = [c for c in required_cols if c not in df_batch.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        st.write(f"Loaded {len(df_batch)} tickets — preview:")
        st.dataframe(df_batch.head(3))

        # 2. Run predictions row by row with a progress bar
        if st.button("Run Batch Prediction"):
            results = []
            bar = st.progress(0, text="Running predictions...")
            
            failed_rows = []
            for i, (_, row) in enumerate(df_batch.iterrows()):
                try:
                    label, conf, prob, _ = predict_priority(
                    impact=row['impact'],
                    urgency=row['urgency'],
                    reassignment_count=row['reassignment_count'],
                    reopen_count=row['reopen_count'],
                    contact_type=row['contact_type'],
                    category=row['category'],
                    subcategory=row['subcategory'],
                    opened_at=row['opened_at'],
                    sys_mod_count=row['sys_mod_count'],
                    notify=row['notify'],
                    )
                    results.append({"predicted_priority": label, "confidence_pct": round(conf, 2)})
                    bar.progress((i + 1) / len(df_batch), text=f"Processing ticket {i+1} of {len(df_batch)}...")
                except Exception as e:
                    failed_rows.append(i)
                    results.append({"predicted_priority": "Error", "confidence_pct": 0})

            # 3. Attach results to df_batch and display
            results_df = pd.DataFrame(results)
            df_batch[["predicted_priority", "confidence_pct"]] = results_df[["predicted_priority", "confidence_pct"]].values

            st.dataframe(df_batch)
            if failed_rows:
                st.caption(f"⚠️ Prediction failed for rows: {failed_rows}. Check input format and values.")
            # 4. Summary
            n_high = (df_batch["predicted_priority"] == "High Priority").sum()
            st.info(f"{n_high} of {len(df_batch)} tickets flagged as High Priority")

            # 5. Download
            st.download_button(
                label="Download Results as CSV",
                data=df_batch.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
