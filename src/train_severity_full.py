import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")

# Load full dataset
df_full = pd.read_csv("data/processed/nvd_cleaned.csv")

# Create mask (IMPORTANT for alignment)
mask = df_full["description"].notna() & (df_full["severity"] != "NONE")

# Apply mask
df = df_full[mask].reset_index(drop=True)

print("Dataset size after cleaning:", len(df))

print("Loading embeddings...")

# Load embeddings
X_full = np.load("data/processed/text_embeddings.npy")

# Apply SAME mask to embeddings (CRITICAL STEP)
X = X_full[mask.values]

# Target labels
y = df["severity"]

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    class_weight={
        "LOW": 2,
        "MEDIUM": 1,
        "HIGH": 1.5,
        "CRITICAL": 2
    }
)

model.fit(X_train, y_train)

print("Evaluating model...")

preds = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, preds)
print("\nAccuracy:", acc)

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, preds, zero_division=0))