import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_training.csv")

print("Loading embeddings...")

X_text = np.load("data/processed/text_embeddings.npy")

# -----------------------------
# Keep top 20 CWE types
# -----------------------------

top_cwe = df["cwe"].value_counts().head(20).index

df = df[df["cwe"].isin(top_cwe)]

X_text = X_text[df.index]

print("Dataset after filtering:", len(df))

# -----------------------------
# Encode labels
# -----------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(df["cwe"])

# -----------------------------
# Train test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------

print("Training CWE classifier...")

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("\nAccuracy:", acc)

print("\nClassification Report:")

print(classification_report(y_test, preds))

# -----------------------------
# Save model
# -----------------------------

with open("models/cwe_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/cwe_label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nCWE model saved.")