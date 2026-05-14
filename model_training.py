import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay

# Load preprocessed splits from feature_engineering.py output
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

# Confirm stratification preserved the ~9.7% minority class in both splits
print(f"\ny_train class ratio:\n{y_train.value_counts(normalize=True)}")
print(f"\ny_test class ratio:\n{y_test.value_counts(normalize=True)}")

# --- Decision Tree ---
# max_depth=5 limits overfitting; balanced weights compensate for 9:1 class imbalance
model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print(export_text(model, feature_names=list(X_train.columns)))
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Normal', 'High Priority']))

disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Normal', 'High Priority'], cmap='Blues')
disp.ax_.set_title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_dt.png")
plt.show()
print("Saved confusion_matrix_dt.png")

# --- Random Forest ---
# 100 trees with shallow depth; n_jobs=-1 uses all CPU cores
rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'High Priority']))

disp_rf = ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, display_labels=['Normal', 'High Priority'], cmap='Greens')
disp_rf.ax_.set_title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.show()
print("Saved confusion_matrix_rf.png")

# predict_proba gives probability scores needed for ROC-AUC (not hard 0/1 predictions)
y_prob_dt = model.predict_proba(X_test)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Compare both models on High Priority class only (pos_label=1)
results = {
    'Model':     ['Decision Tree', 'Random Forest'],
    'Precision': [precision_score(y_test, y_pred,    pos_label=1),
                  precision_score(y_test, y_pred_rf, pos_label=1)],
    'Recall':    [recall_score(y_test, y_pred,    pos_label=1),
                  recall_score(y_test, y_pred_rf, pos_label=1)],
    'F1':        [f1_score(y_test, y_pred,    pos_label=1),
                  f1_score(y_test, y_pred_rf, pos_label=1)],
    'ROC-AUC':   [roc_auc_score(y_test, y_prob_dt),
                  roc_auc_score(y_test, y_prob_rf)],
}

df_results = pd.DataFrame(results).set_index('Model').round(4)
print("\nModel Comparison (High Priority class):")
print(df_results)

# Rank features by how much they reduce impurity across all trees
df_importance = pd.DataFrame({
    'Feature':    X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(df_importance['Feature'], df_importance['Importance'])
plt.gca().invert_yaxis()  # highest importance at top
plt.xlabel('Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()
print("Saved feature_importance_rf.png")

# --- ROC Curve Comparison ---
# Overlay both models on one axes; from_estimator uses predict_proba internally
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(model,    X_test, y_test, ax=ax, name='Decision Tree')
RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, name='Random Forest')

# Diagonal dashed line = random classifier baseline (AUC = 0.5)
ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
ax.set_title("ROC Curve Comparison")
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve_comparison.png")
plt.show()
print("Saved roc_curve_comparison.png")

# --- Model Persistence ---
# Serialize the trained Random Forest so it can be reloaded without retraining
joblib.dump(rf_model, 'best_model_day3.joblib')
print("Model saved as best_model_day3.joblib")

# Reload from disk and verify it produces identical predictions
loaded_model = joblib.load('best_model_day3.joblib')

sample = X_test.iloc[[0]]
pred_inmemory = rf_model.predict(sample)[0]
pred_loaded   = loaded_model.predict(sample)[0]

print(f"In-memory prediction:  {pred_inmemory}")
print(f"Loaded model prediction: {pred_loaded}")
print(f"Predictions match: {pred_inmemory == pred_loaded}")
