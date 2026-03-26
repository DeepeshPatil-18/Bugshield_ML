import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_cleaned.csv")

# Clean same as before
df = df[df["description"].notna()]
df = df[df["severity"] != "NONE"]
df = df.reset_index(drop=True)

print("Dataset size:", len(df))

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
print("Loading embeddings...")

embeddings = np.load("data/processed/text_embeddings.npy")

# Align
embeddings = embeddings[:len(df)]

# Convert to float32 (REQUIRED for FAISS)
embeddings = embeddings.astype("float32")

print("Embeddings shape:", embeddings.shape)

# -----------------------------
# BUILD FAISS INDEX
# -----------------------------
print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors")

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

    query_embedding = model.encode([query]).astype("float32")

    print("Searching with FAISS...")

    distances, indices = index.search(query_embedding, top_k)

    print("\nTop similar vulnerabilities:\n")

    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}")
        print("CVE ID:", df.iloc[idx]["cve_id"])
        print("Severity:", df.iloc[idx]["severity"])
        print("CVSS:", df.iloc[idx]["cvss_score"])
        print("Description:", df.iloc[idx]["description"][:200])
        print("Distance:", distances[0][i])
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