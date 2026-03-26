import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

print("Loading dataset...")

# Load dataset
df_full = pd.read_csv("data/processed/nvd_cleaned.csv")

# Create mask (IMPORTANT for alignment)
mask = df_full["description"].notna() & (df_full["severity"] != "NONE")

# Apply mask
df = df_full[mask].reset_index(drop=True)

print("Dataset size:", len(df))

print("Loading embeddings...")

# Load embeddings
X_text_full = np.load("data/processed/text_embeddings.npy")

# Apply SAME mask to embeddings (CRITICAL STEP)
X_text = X_text_full[mask.values]

# -----------------------------
# STRUCTURED FEATURES
# -----------------------------
print("Processing structured features...")

structured_cols = [
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction"
]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

X_struct = encoder.fit_transform(df[structured_cols])

# -----------------------------
# TARGET
# -----------------------------
y = df["cvss_score"]

# -----------------------------
# COMBINE FEATURES
# -----------------------------
print("Combining features...")

X = np.hstack([X_text, X_struct])

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODEL (FAST + EFFICIENT)
# -----------------------------
print("Training model...")

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
print("Evaluating...")

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)