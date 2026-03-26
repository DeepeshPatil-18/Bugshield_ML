import pandas as pd

INPUT = "data/processed/nvd_cleaned.csv"
OUTPUT = "data/processed/nvd_training.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT)

print("Original size:", len(df))

# remove NONE severity
df = df[df["severity"] != "NONE"]

print("After removing NONE:", len(df))

# balance dataset
min_samples = df["severity"].value_counts().min()

balanced_df = (
    df.groupby("severity", group_keys=False)
    .sample(min_samples, random_state=42)
)

print("\nBalanced distribution:")
print(balanced_df["severity"].value_counts())

balanced_df.to_csv(OUTPUT, index=False)

print("\nTraining dataset saved:", OUTPUT)