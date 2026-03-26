import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_cleaned.csv")
df = df.dropna(subset=["description"]).reset_index(drop=True)

print("Dataset size:", len(df))

model = SentenceTransformer("all-MiniLM-L6-v2")

# File to store embeddings
save_path = "data/processed/text_embeddings.npy"

# Check if partial file exists
if os.path.exists(save_path):
    print("Resuming from saved embeddings...")
    existing_embeddings = np.load(save_path)
    start_idx = len(existing_embeddings)
else:
    existing_embeddings = None
    start_idx = 0

batch_size = 256

all_embeddings = []

if existing_embeddings is not None:
    all_embeddings.append(existing_embeddings)

print(f"Starting from index: {start_idx}")

for i in range(start_idx, len(df), batch_size):
    batch_texts = df["description"].iloc[i:i+batch_size].tolist()

    embeddings = model.encode(batch_texts)

    all_embeddings.append(embeddings)

    # Save progress after each batch
    combined = np.vstack(all_embeddings)
    np.save(save_path, combined)

    print(f"Saved up to index: {i + batch_size}")

print("All embeddings generated and saved successfully.")