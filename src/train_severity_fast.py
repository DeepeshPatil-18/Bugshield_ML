import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_training.csv")

print("Loading cached embeddings...")

text_embeddings = np.load("data/processed/text_embeddings.npy")

print("Embeddings shape:", text_embeddings.shape)

# Structured features
structured_cols = [
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction"
]

encoder = OneHotEncoder()

structured_features = encoder.fit_transform(
    df[structured_cols]
).toarray()

# Combine features
X = np.hstack([
    text_embeddings,
    structured_features
])

y = df["severity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVM": LinearSVC(),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )
}

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    print(classification_report(y_test, preds))

    results.append((name, acc))

best_model_name, best_acc = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)[0]

print("\nBest model:", best_model_name)

best_model = models[best_model_name]

with open("models/severity_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved.")