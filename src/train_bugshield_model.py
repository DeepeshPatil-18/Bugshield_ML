import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("Loading training dataset...")

df = pd.read_csv("data/processed/nvd_training.csv")

print("Dataset size:", len(df))


# -------------------------
# TEXT EMBEDDINGS
# -------------------------

print("Loading sentence transformer...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

text_embeddings = embedder.encode(
    df["description"].tolist(),
    batch_size=64,
    show_progress_bar=True
)


# -------------------------
# STRUCTURED FEATURES
# -------------------------

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


# -------------------------
# CVSS SCORE FEATURE
# -------------------------

cvss_scores = df["cvss_score"].values.reshape(-1,1)


# -------------------------
# COMBINE FEATURES
# -------------------------

X = np.hstack([
    text_embeddings,
    structured_features,
    cvss_scores
])

y = df["severity"]


# -------------------------
# SCALE FEATURES
# -------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)


# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -------------------------
# TRAIN MODEL
# -------------------------

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=2000
)

model.fit(X_train, y_train)


# -------------------------
# EVALUATE
# -------------------------

print("Evaluating model...")

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, preds))


# -------------------------
# SAVE MODELS
# -------------------------

print("Saving models...")

with open("models/bugshield_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/embedding_model.pkl", "wb") as f:
    pickle.dump(embedder, f)

with open("models/feature_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Models saved.")