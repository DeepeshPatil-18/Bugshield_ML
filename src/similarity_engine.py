import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_cleaned.csv")

# Clean same as training
df = df[df["description"].notna()]
df = df[df["severity"] != "NONE"]
df = df.reset_index(drop=True)

print("Dataset size:", len(df))

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
print("Loading embeddings...")

embeddings = np.load("data/processed/text_embeddings.npy")

# Align (IMPORTANT)
embeddings = embeddings[:len(df)]

print("Embeddings shape:", embeddings.shape)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_similar(query, top_k=5):
    print("\nEncoding query...")

    query_embedding = model.encode([query])

    print("Computing similarity...")

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top results
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print("\nTop similar vulnerabilities:\n")

    for i, idx in enumerate(top_indices):
        print(f"Rank {i+1}")
        print("CVE ID:", df.iloc[idx]["cve_id"])
        print("Severity:", df.iloc[idx]["severity"])
        print("CVSS:", df.iloc[idx]["cvss_score"])
        print("Description:", df.iloc[idx]["description"][:200])
        print("-" * 80)


# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nEnter vulnerability description (or 'exit'): ")

        if query.lower() == "exit":
            break

        search_similar(query, top_k=5)