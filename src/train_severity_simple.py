import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/processed/nvd_cleaned.csv")

# Clean data
df = df[df["description"].notna()]
df = df[df["severity"] != "NONE"]
df = df.reset_index(drop=True)

print("Dataset size:", len(df))

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
print("Loading embeddings...")

embeddings = np.load("data/processed/text_embeddings.npy")

# Align embeddings with dataset
embeddings = embeddings[:len(df)]

print("Embeddings shape:", embeddings.shape)

# -----------------------------
# TARGET
# -----------------------------
y = df["severity"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
print("Evaluating model...")

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("\nAccuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# -----------------------------
# SAVE MODEL
# -----------------------------
print("Saving model...")

with open("models/severity_model_simple.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")