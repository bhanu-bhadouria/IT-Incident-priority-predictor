import pandas as pd

df = pd.read_csv("incidents_day1.csv")
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