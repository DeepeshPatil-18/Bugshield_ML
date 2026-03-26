import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer


DATA_PATH = "data/processed/cleaned_cve.csv"

MODEL_PATH = "models/severity_model.pkl"
EMBED_MODEL_PATH = "models/embedding_model.pkl"


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset size:", len(df))

print("\nClass distribution:")
print(df["severity"].value_counts())


# -----------------------------
# BALANCE DATASET
# -----------------------------

min_samples = df["severity"].value_counts().min()

balanced_df = (
    df.groupby("severity", group_keys=False)
    .sample(n=min_samples, random_state=42)
)

print("\nBalanced dataset size:", len(balanced_df))


X = balanced_df["description"]
y = balanced_df["severity"]


# -----------------------------
# LOAD SENTENCE TRANSFORMER
# -----------------------------

print("\nLoading embedding model...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")


print("\nGenerating embeddings (this may take a few minutes)...")

X_embeddings = embedder.encode(
    X.tolist(),
    show_progress_bar=True
)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -----------------------------
# TRAIN CLASSIFIER
# -----------------------------

print("\nTraining classifier...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# -----------------------------
# EVALUATE
# -----------------------------

print("\nEvaluating model...")

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, preds))


# -----------------------------
# SAVE MODELS
# -----------------------------

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(EMBED_MODEL_PATH, "wb") as f:
    pickle.dump(embedder, f)

print("\nModels saved to models/")