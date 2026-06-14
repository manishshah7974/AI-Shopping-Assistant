import os
import json
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv
from embedding import EmbeddingModel
from data_loader import load_products
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "")

INDEX_SCHEMA = {
    "index": {"name": "products", "prefix": "product"},
    "fields": [
        {"name": "name", "type": "text"},
        {"name": "brandName", "type": "text"},
        {"name": "sellingPrice", "type": "numeric"},
        {"name": "discountPercentage", "type": "numeric"},
        {"name": "inventory", "type": "numeric"},
        {"name": "gender", "type": "text"},
        {"name": "categoryName", "type": "text"},
        {"name": "attributeSetName", "type": "text"},
        {"name": "slug", "type": "text"},
        {"name": "sku", "type": "text"},
        {"name": "product_idx", "type": "numeric"},
        {"name": "metadata_json", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 384,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ],
}


class VectorStore:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.index = SearchIndex.from_dict(INDEX_SCHEMA, redis_url=REDIS_URL)

    def build_index(self, json_path: str):
        """Load products, embed them, and store in Redis."""
        documents = load_products(json_path)
        embeddings = self.embedding_model.embed_products(documents)
        metadata = [doc["metadata"] for doc in documents]

        # Create the Redis index
        self.index.create(overwrite=True)

        # Load data in batches
        batch_size = 500
        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = []
            for i in range(start, end):
                m = metadata[i]
                record = {
                    "name": m.get("name", ""),
                    "brandName": m.get("brandName", ""),
                    "sellingPrice": m.get("sellingPrice", 0) or 0,
                    "discountPercentage": m.get("discountPercentage", 0) or 0,
                    "inventory": m.get("inventory", 0) or 0,
                    "gender": ", ".join(m.get("gender", [])),
                    "categoryName": ", ".join(m.get("categoryName", [])),
                    "attributeSetName": m.get("attributeSetName", ""),
                    "slug": m.get("slug", ""),
                    "sku": m.get("sku", ""),
                    "product_idx": i,
                    "metadata_json": json.dumps(m),
                    "embedding": embeddings[i].tobytes(),
                }
                batch.append(record)
            self.index.load(batch)
            print(f"[INFO] Loaded {end}/{total} products into Redis")

        print(f"[INFO] Redis vector index built with {total} vectors")

    def load(self):
        """Verify Redis index exists."""
        if not self.index.exists():
            raise RuntimeError("Redis index 'products' not found. Run build_index first.")
        info = self.index.info()
        num_docs = int(info.get("num_docs", 0))
        print(f"[INFO] Connected to Redis index ({num_docs} vectors)")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar products by query text."""
        query_emb = self.embedding_model.embed_query(query)
        q = VectorQuery(
            vector=query_emb[0].tolist(),
            vector_field_name="embedding",
            return_fields=["metadata_json", "product_idx"],
            num_results=top_k,
        )
        raw_results = self.index.query(q)
        results = []
        for doc in raw_results:
            meta = json.loads(doc["metadata_json"])
            score = float(doc.get("vector_distance", 0))
            results.append({"score": score, "metadata": meta, "product_idx": int(doc.get("product_idx", -1))})
        return results

    def get_embedding_by_idx(self, product_idx: int) -> Optional[np.ndarray]:
        """Fetch a product's raw embedding from Redis by its index."""
        import redis as r
        client = r.from_url(REDIS_URL)
        keys = client.keys("product:*")
        for key in keys:
            idx = client.hget(key, "product_idx")
            if idx is not None and int(idx) == product_idx:
                emb_bytes = client.hget(key, "embedding")
                if emb_bytes:
                    return np.frombuffer(emb_bytes, dtype=np.float32)
        return None

    def search_by_vector(self, vector: List[float], top_k: int = 10, exclude_name: str = "") -> List[Dict]:
        """Find similar products given a vector as a list of floats."""
        q = VectorQuery(
            vector=vector,
            vector_field_name="embedding",
            return_fields=["metadata_json"],
            num_results=top_k + 1,
        )
        raw_results = self.index.query(q)
        results = []
        for doc in raw_results:
            meta = json.loads(doc["metadata_json"])
            if exclude_name and meta["name"].lower() == exclude_name.lower():
                continue
            score = float(doc.get("vector_distance", 0))
            results.append({"score": score, "metadata": meta})
        return results[:top_k]


# Quick test
if __name__ == "__main__":
    store = VectorStore()

    if not store.index.exists():
        store.build_index("Fragrance_and_Beauty_cleaned.json")
    else:
        store.load()

    results = store.search("vanilla perfume for women under 20", top_k=3)
    print("\n--- Search: 'vanilla perfume for women under 20' ---")
    for r in results:
        m = r["metadata"]
        print(f"  {m['name']} | {m['brandName']} | {m['sellingPrice']} | {m['gender']}")
