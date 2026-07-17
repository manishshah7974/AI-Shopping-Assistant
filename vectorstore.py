import os
import json
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv
from embedding import EmbeddingModel
from data_loader import load_products, build_product_text
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
        """Load products, embed them, and store in Redis with SKU-based keys."""
        documents = load_products(json_path)
        embeddings = self.embedding_model.embed_products(documents)
        metadata = [doc["metadata"] for doc in documents]

        # Create the Redis index
        self.index.create(overwrite=True)

        # Load data in batches using SKU as the key
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
                    "metadata_json": json.dumps(m),
                    "embedding": embeddings[i].tobytes(),
                }
                batch.append(record)
            # Use SKU as the id_field so keys become product:<sku>
            self.index.load(batch, id_field="sku")
            print(f"[INFO] Loaded {end}/{total} products into Redis")

        print(f"[INFO] Redis vector index built with {total} vectors (SKU-based keys)")

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
            return_fields=["metadata_json", "sku"],
            num_results=top_k,
        )
        raw_results = self.index.query(q)
        results = []
        for doc in raw_results:
            meta = json.loads(doc["metadata_json"])
            score = float(doc.get("vector_distance", 0))
            results.append({
                "score": score,
                "metadata": meta,
                "sku": doc.get("sku", ""),
            })
        return results

    def get_embedding_by_sku(self, sku: str) -> Optional[np.ndarray]:
        """Fetch a product's raw embedding from Redis by SKU (direct key lookup)."""
        record = self.index.fetch(sku)
        if record and "embedding" in record:
            emb_bytes = record["embedding"]
            if isinstance(emb_bytes, bytes):
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

    # ─── CRUD Operations (SKU-based) ───

    def update_product(self, sku: str, updated_product: dict):
        """Re-embed and update a single product in Redis by SKU.

        Args:
            sku: The product SKU (e.g., "I-00000028599")
            updated_product: Full product dict with all fields (name, brandName, etc.)
        """
        # Rebuild the text and re-embed
        text = build_product_text(updated_product)
        embedding = self.embedding_model.embed_query(text)[0]

        # Build updated record
        record = {
            "name": updated_product.get("name", ""),
            "brandName": updated_product.get("brandName", ""),
            "sellingPrice": updated_product.get("sellingPrice", 0) or 0,
            "discountPercentage": updated_product.get("discountPercentage", 0) or 0,
            "inventory": updated_product.get("inventory", 0) or 0,
            "gender": ", ".join(updated_product.get("gender", [])),
            "categoryName": ", ".join(updated_product.get("categoryName", [])),
            "attributeSetName": updated_product.get("attributeSetName", ""),
            "slug": updated_product.get("slug", ""),
            "sku": sku,
            "metadata_json": json.dumps(updated_product),
            "embedding": embedding.tobytes(),
        }

        # Load with id_field="sku" overwrites the existing key product:<sku>
        self.index.load([record], id_field="sku")
        print(f"[INFO] Updated product:{sku} — {updated_product.get('name')}")

    def update_metadata_only(self, sku: str, updated_product: dict):
        """Update metadata fields without re-embedding (use when only price/stock changed).

        Args:
            sku: The product SKU
            updated_product: Full product dict with updated metadata
        """
        import redis as r

        existing = self.index.fetch(sku)
        if not existing:
            print(f"[ERROR] Product not found: product:{sku}")
            return

        client = r.from_url(REDIS_URL)
        key = f"product:{sku}"

        # Update only metadata fields, keep embedding unchanged
        client.hset(key, mapping={
            "name": updated_product.get("name", ""),
            "brandName": updated_product.get("brandName", ""),
            "sellingPrice": updated_product.get("sellingPrice", 0) or 0,
            "discountPercentage": updated_product.get("discountPercentage", 0) or 0,
            "inventory": updated_product.get("inventory", 0) or 0,
            "gender": ", ".join(updated_product.get("gender", [])),
            "categoryName": ", ".join(updated_product.get("categoryName", [])),
            "attributeSetName": updated_product.get("attributeSetName", ""),
            "slug": updated_product.get("slug", ""),
            "metadata_json": json.dumps(updated_product),
        })
        print(f"[INFO] Updated metadata for product:{sku} (no re-embedding)")

    def delete_product(self, sku: str):
        """Delete a product from Redis by SKU.

        Args:
            sku: The product SKU (e.g., "I-00000028599")
        """
        # drop_documents expects the document ID (without prefix)
        deleted = self.index.drop_documents(sku)
        if deleted:
            print(f"[INFO] Deleted product:{sku}")
        else:
            print(f"[WARN] Product not found: product:{sku}")


# Quick test
if __name__ == "__main__":
    store = VectorStore()

    if store.index.exists():
        info = store.index.info()
        num_docs = int(info.get("num_docs", 0))
        if num_docs == 0:
            print("[WARN] Index exists but is empty. Rebuilding...")
            store.build_index("Fragrance_and_Beauty_cleaned.json")
        else:
            store.load()
    else:
        store.build_index("Fragrance_and_Beauty_cleaned.json")

    results = store.search("vanilla perfume for women under 20", top_k=3)
    print("\n--- Search: 'vanilla perfume for women under 20' ---")
    for r in results:
        m = r["metadata"]
        print(f"  {m['name']} | {m['brandName']} | {m['sellingPrice']} | SKU: {r['sku']}")
