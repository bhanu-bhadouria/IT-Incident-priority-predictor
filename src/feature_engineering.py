import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/raw/incidents_day1.csv")
print(df.shape)
print(df.dtypes)
print(df.head(3))

print(df['location'].value_counts().head(5))
df['location'] = df['location'].fillna('Unknown')
df['assignment_group'] = df['assignment_group'].fillna('Unassigned')
print(df.isna().sum())
df['opened_at'] = pd.to_datetime(df['opened_at'])

hour = df['opened_at'].dt.hour
day_of_week = df['opened_at'].dt.dayofweek  # Monday=0, Sunday=6

df['is_business_hours'] = ((hour >= 9) & (hour < 17) & (day_of_week < 5)).astype(int)
df['is_weekend'] = (day_of_week >= 5).astype(int)
df['is_night'] = ((hour < 8) | (hour >= 18)).astype(int)
df['quarter'] = df['opened_at'].dt.quarter

print(df[['opened_at', 'is_business_hours', 'is_weekend', 'is_night', 'quarter']].head(10))

# Severity hint: mirrors ServiceNow's impact/urgency matrix logic
df['impact_urgency_score'] = df['impact'] + df['urgency']

# Tickets bounced across teams AND reopened = strong high-priority signal
df['escalation_risk'] = ((df['reassignment_count'] >= 2) & (df['reopen_count'] >= 1)).astype(int)

print(df[['impact', 'urgency', 'impact_urgency_score', 'reassignment_count', 'reopen_count', 'escalation_risk']].head(10))
print("\nescalation_risk distribution:")
print(df['escalation_risk'].value_counts())

print(df.groupby('is_high_priority')['impact_urgency_score'].mean())

label_encoders = {}
for col in ['contact_type', 'category', 'subcategory','location', 'assignment_group']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"{col} classes: {le.classes_}")

# Step 6: Class imbalance analysis
print(df['is_high_priority'].value_counts())
print(df['is_high_priority'].value_counts(normalize=True))

n = len(df)
n_class_1 = df['is_high_priority'].sum()
weight_class_1 = n / (2 * n_class_1)
print(f"Manual class_weight for class 1: {weight_class_1:.4f}")

drop_cols = ['is_high_priority', 'priority', 'priority_raw', 'number', 'opened_at', 'caller_id', 'opened_by',
             'made_sla', 'incident_state', 'u_priority_confirmation', 'sys_mod_count', 'active', 'knowledge', 'problem_id', 'cmdb_ci', 'sys_created_by', 'sys_created_at', 'sys_updated_by', 'sys_updated_at']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['is_high_priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"y_train class ratio:\n{y_train.value_counts(normalize=True)}")
print(f"y_test class ratio:\n{y_test.value_counts(normalize=True)}")
print(X.columns.tolist())

X_train.to_csv("../data/processed/X_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)
joblib.dump(label_encoders, "../models/label_encoders.joblib")
print("All files saved.")
