import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"[INFO] Loaded embedding model: {model_name} (dim={self.dimension})")

    def embed_products(self, documents: List[Dict], batch_size: int = 256) -> np.ndarray:
        """
        Embed the 'text' field of each product document.
        Returns numpy array of shape (n_products, embedding_dim).
        """
        texts = [doc["text"] for doc in documents]
        print(f"[INFO] Embedding {len(texts)} products...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode([query]).astype("float32")


# Quick test
if __name__ == "__main__":
    from data_loader import load_products

    docs = load_products("Fragrance_and_Beauty_cleaned.json")
    emb = EmbeddingModel()

    # Test with first 5 products
    sample_embeddings = emb.embed_products(docs[:5])
    print(f"\nSample embedding (first 10 dims): {sample_embeddings[0][:10]}")

    # Test query embedding
    q = emb.embed_query("vanilla perfume for women")
    print(f"Query embedding shape: {q.shape}")
