import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# LOAD DATA
# -----------------------------

print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_training.csv")

print("Dataset size:", len(df))

# -----------------------------
# TEXT EMBEDDINGS
# -----------------------------

print("Loading sentence transformer...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

text_embeddings = embedder.encode(
    df["description"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# -----------------------------
# STRUCTURED FEATURES
# -----------------------------

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

# -----------------------------
# COMBINE FEATURES
# -----------------------------

X = np.hstack([
    text_embeddings,
    structured_features
])

y = df["severity"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# MODELS
# -----------------------------

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

# -----------------------------
# TRAIN + EVALUATE
# -----------------------------

for name, model in models.items():

    print("\n==============================")
    print("Training:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    print(classification_report(y_test, preds))

    results.append((name, acc))

# -----------------------------
# SAVE BEST MODEL
# -----------------------------

best_model_name, best_acc = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)[0]

print("\nBest model:", best_model_name)

best_model = models[best_model_name]

with open("models/severity_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/embedding_model.pkl", "wb") as f:
    pickle.dump(embedder, f)

with open("models/feature_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Models saved.")