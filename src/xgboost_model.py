import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, PrecisionRecallDisplay

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

print("""
Baseline Results (High Priority class):
Model         | Precision | Recall | F1   | AUC
--------------+-----------+--------+------+------
Decision Tree | 0.50      | 0.99   | 0.67 | 0.99
Random Forest | 0.39      | 1.00   | 0.56 | 0.99
""")

# scale_pos_weight tells XGBoost how much to penalise missing a positive (High Priority) sample
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Hold out 20% of training data for early stopping — keeps test set unseen during training
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# early_stopping_rounds halts training if aucpr doesn't improve for 20 consecutive rounds
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=20,
    eval_metric='aucpr',
    random_state=42
)

xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Normal', 'High Priority']))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test, display_labels=['Normal', 'High Priority'], cmap='Oranges')
disp.ax_.set_title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig("../outputs/plots/confusion_matrix_xgb.png")
plt.show()
print("Saved confusion_matrix_xgb.png")

# 3-model comparison — DT and RF metrics from Day 3, XGBoost calculated fresh
results = {
    'Model':     ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Precision': [0.50, 0.39, precision_score(y_test, y_pred, pos_label=1)],
    'Recall':    [0.99, 1.00, recall_score(y_test,    y_pred, pos_label=1)],
    'F1':        [0.67, 0.56, f1_score(y_test,        y_pred, pos_label=1)],
    'ROC-AUC':   [0.99, 0.99, roc_auc_score(y_test,   y_prob)],
}

df_results = pd.DataFrame(results).set_index('Model').round(4)
print("\nModel Comparison (High Priority class):")
print(df_results)

# --- Threshold Tuning ---
# Default threshold is 0.5; lowering it catches more High Priority but increases false alarms
print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1':<10}")
print("-" * 44)
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_t = (y_prob >= threshold).astype(int)
    p = precision_score(y_test, y_pred_t, pos_label=1)
    r = recall_score(y_test,    y_pred_t, pos_label=1)
    f = f1_score(y_test,        y_pred_t, pos_label=1)
    print(f"{threshold:<12} {p:<12.4f} {r:<10.4f} {f:<10.4f}")

# --- Precision-Recall Curve ---
# Shows the full precision/recall tradeoff across all thresholds — better than ROC for imbalanced data
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(xgb, X_test, y_test, ax=ax, name='XGBoost')
ax.set_title("Precision-Recall Curve — XGBoost")
plt.tight_layout()
plt.savefig("../outputs/plots/precision_recall_xgb.png")
plt.show()
print("Saved precision_recall_xgb.png")

# shap.Explainer returns an Explanation object required by the new plot API
explainer = shap.Explainer(xgb, X_train)
shap_values = explainer(X_test)

# Summary plot — shows which features most influence predictions across all test samples
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../outputs/plots/shap_summary.png")
plt.show()
print("Saved shap_summary.png")

# Waterfall plot for one High Priority sample — shows per-feature contribution for that prediction
hp_idx = int((y_test == 1).values.nonzero()[0][0])
shap.plots.waterfall(shap_values[hp_idx], show=False)
plt.tight_layout()
plt.savefig("../outputs/plots/shap_waterfall.png")
plt.show()
print("Saved shap_waterfall.png")

from sklearn.inspection import permutation_importance

perm = permutation_importance(xgb, X_test, y_test, n_repeats=10, random_state=42, scoring='f1')

perm_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm.importances_mean
}).sort_values('importance', ascending=False)

print(perm_df)
plt.figure(figsize=(10, 6))

# --- Save model artefacts ---
joblib.dump(xgb,                          'best_model_final.joblib')
joblib.dump(list(X_train.columns),        'feature_list.joblib')
joblib.dump(0.7,                          'threshold.joblib')
print("Saved best_model_final.joblib, feature_list.joblib, threshold.joblib")