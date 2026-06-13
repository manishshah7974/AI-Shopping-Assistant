import os
import json
import numpy as np
from typing import List, Dict
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
        self.metadata = []

    def build_index(self, json_path: str):
        """Load products, embed them, and store in Redis."""
        documents = load_products(json_path)
        embeddings = self.embedding_model.embed_products(documents)
        self.metadata = [doc["metadata"] for doc in documents]

        # Create the Redis index
        self.index.create(overwrite=True)

        # Load data in batches
        batch_size = 500
        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = []
            for i in range(start, end):
                m = self.metadata[i]
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
        """Verify Redis index exists and load metadata."""
        if not self.index.exists():
            raise RuntimeError("Redis index 'products' not found. Run build_index first.")
        info = self.index.info()
        num_docs = int(info.get("num_docs", 0))
        print(f"[INFO] Connected to Redis index ({num_docs} vectors)")
        # Load metadata from Redis (rebuild from stored JSON)
        self._load_metadata()

    def _load_metadata(self):
        """Load all metadata from Redis into memory for search_by_index."""
        import redis as r
        client = r.from_url(REDIS_URL)
        keys = sorted(client.keys("product:*"), key=lambda k: k.decode())
        self.metadata = []
        for key in keys:
            data = client.hgetall(key)
            if b"metadata_json" in data:
                self.metadata.append(json.loads(data[b"metadata_json"]))
        print(f"[INFO] Loaded {len(self.metadata)} metadata records")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar products by query text."""
        query_emb = self.embedding_model.embed_query(query)
        q = VectorQuery(
            vector=query_emb[0].tolist(),
            vector_field_name="embedding",
            return_fields=["metadata_json"],
            num_results=top_k,
        )
        raw_results = self.index.query(q)
        results = []
        for doc in raw_results:
            meta = json.loads(doc["metadata_json"])
            score = float(doc.get("vector_distance", 0))
            results.append({"score": score, "metadata": meta})
        return results

    def search_by_index(self, product_idx: int, top_k: int = 10) -> List[Dict]:
        """Find similar products given a product index."""
        if product_idx >= len(self.metadata):
            return []
        # Use the product's name as a search query to find its vector
        product_name = self.metadata[product_idx]["name"]
        # Get the product's embedding by searching for it
        import redis as r
        client = r.from_url(REDIS_URL)
        keys = client.keys("product:*")
        # Find the key with matching product_idx
        target_embedding = None
        for key in keys:
            data = client.hgetall(key)
            if b"product_idx" in data:
                idx = int(data[b"product_idx"])
                if idx == product_idx:
                    target_embedding = np.frombuffer(data[b"embedding"], dtype=np.float32)
                    break

        if target_embedding is None:
            # Fallback: embed the product name
            target_embedding = self.embedding_model.embed_query(product_name)[0]

        q = VectorQuery(
            vector=target_embedding.tolist(),
            vector_field_name="embedding",
            return_fields=["metadata_json", "product_idx"],
            num_results=top_k + 1,
        )
        raw_results = self.index.query(q)
        results = []
        for doc in raw_results:
            doc_idx = int(doc.get("product_idx", -1))
            if doc_idx == product_idx:
                continue
            meta = json.loads(doc["metadata_json"])
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
