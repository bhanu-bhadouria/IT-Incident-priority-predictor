import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/incidents_day1.csv")

# ── Step 1: First Look ────────────────────────────────────────────────────────
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nInfo:")
df.info()

# ── Step 2: Missing Values ────────────────────────────────────────────────────
print("\nMissing value counts:")
print(df.isnull().sum())
print("\nMissing value percentages:")
print((df.isnull().mean() * 100).round(2))

# ── Step 3: Target Variable Distribution ─────────────────────────────────────
print("\nPriority distribution (counts):")
print(df["priority"].value_counts())
print("\nPriority distribution (proportions):")
print(df["priority"].value_counts(normalize=True).round(3))

print("\nis_high_priority distribution:")
print(df["is_high_priority"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["priority"].value_counts().sort_index().plot(kind="bar", ax=axes[0], title="Priority Distribution")
df["is_high_priority"].value_counts().plot(kind="bar", ax=axes[1], title="is_high_priority Distribution")
plt.tight_layout()
plt.savefig("../outputs/plots/target_distribution.png")
plt.show()

# ── Step 4: Numerical Features ────────────────────────────────────────────────
print("\nNumerical summary:")
print(df.describe())

num_cols = ["reassignment_count", "reopen_count", "sys_mod_count", "impact", "urgency"]
fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 4))
for ax, col in zip(axes, num_cols):
    df[col].hist(bins=30, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("../outputs/plots/numerical_distributions.png")
plt.show()

# ── Step 5: Categorical Features ─────────────────────────────────────────────
cat_cols = ["category", "contact_type", "incident_state"]
for col in cat_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# ── Step 6: Relationships with Target ─────────────────────────────────────────
print("\nMean reassignment_count by priority:")
print(df.groupby("priority")["reassignment_count"].mean().round(2))
print("\nMean urgency by priority:")
print(df.groupby("priority")["urgency"].mean().round(2))

cross = pd.crosstab(df["category"], df["priority"], normalize="index").round(3)
print("\nCategory vs Priority (row-normalised):")
print(cross)

plt.figure(figsize=(12, 6))
sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues")
plt.title("Category vs Priority (proportion)")
plt.tight_layout()
plt.savefig("../outputs/plots/category_vs_priority.png")
plt.show()

# ── Step 7: Time-Based Patterns ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df.groupby("hour")["is_high_priority"].mean().plot(ax=axes[0], marker="o", title="High Priority Rate by Hour")
df.groupby("day_of_week")["priority"].mean().plot(ax=axes[1], marker="o", title="Avg Priority by Day of Week")
plt.tight_layout()
plt.savefig("../outputs/plots/time_patterns.png")
plt.show()

# ── Step 8: Correlation Matrix ────────────────────────────────────────────────
numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("../outputs/plots/correlation_matrix.png")
plt.show()
