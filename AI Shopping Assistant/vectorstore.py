import os
import faiss
import numpy as np
import pickle
from typing import List, Dict
from embedding import EmbeddingModel
from data_loader import load_products


class VectorStore:
    def __init__(self, persist_dir: str = "faiss_index"):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []  # Full product metadata for each vector
        self.embedding_model = EmbeddingModel()

    def build_index(self, json_path: str):
        """Load products, embed them, and build FAISS index."""
        documents = load_products(json_path)
        embeddings = self.embedding_model.embed_products(documents)
        self.metadata = [doc["metadata"] for doc in documents]

        # Build FAISS index (L2 distance)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors (dim={dim})")

        self.save()

    def save(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved index and metadata to {self.persist_dir}/")

    def load(self):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(os.path.join(self.persist_dir, "index.faiss"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded index ({self.index.ntotal} vectors) from {self.persist_dir}/")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar products by query text."""
        query_emb = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append({"score": float(dist), "metadata": self.metadata[idx]})
        return results

    def search_by_index(self, product_idx: int, top_k: int = 10) -> List[Dict]:
        """Find similar products given a product index (for 'find similar' feature)."""
        vec = self.index.reconstruct(product_idx).reshape(1, -1)
        distances, indices = self.index.search(vec, top_k + 1)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != product_idx and idx < len(self.metadata):
                results.append({"score": float(dist), "metadata": self.metadata[idx]})
        return results[:top_k]


# Quick test
if __name__ == "__main__":
    store = VectorStore()

    # Build or load
    if os.path.exists(os.path.join(store.persist_dir, "index.faiss")):
        store.load()
    else:
        store.build_index("Fragrance_and_Beauty_cleaned.json")

    # Test search
    results = store.search("vanilla perfume for women under 20", top_k=3)
    print("\n--- Search: 'vanilla perfume for women under 20' ---")
    for r in results:
        m = r["metadata"]
        print(f"  {m['name']} | {m['brandName']} | {m['sellingPrice']} | {m['gender']}")
