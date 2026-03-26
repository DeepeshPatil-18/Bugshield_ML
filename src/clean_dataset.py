import pandas as pd

INPUT_FILE = "data/processed/nvd_dataset.csv"
OUTPUT_FILE = "data/processed/nvd_cleaned.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset size:", len(df))

# remove rows with missing description
df = df[df["description"].notna()]

# remove very short descriptions
df = df[df["description"].str.len() > 40]

# remove rows without severity
df = df[df["severity"].notna()]

# remove useless CWE values
df = df[df["cwe"].notna()]
df = df[~df["cwe"].str.contains("noinfo", case=False, na=False)]
df = df[~df["cwe"].str.contains("other", case=False, na=False)]

print("After cleaning:", len(df))

# normalize text
df["description"] = df["description"].str.lower()

# drop duplicates
df = df.drop_duplicates(subset=["description"])

print("After removing duplicates:", len(df))

df.to_csv(OUTPUT_FILE, index=False)

print("Clean dataset saved to:", OUTPUT_FILE)