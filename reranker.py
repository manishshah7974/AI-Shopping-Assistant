"""Cross-Encoder Reranker for two-stage retrieval.

Stage 1 (bi-encoder): Fast approximate retrieval via Redis vector search.
Stage 2 (cross-encoder): Precise reranking of candidates by scoring
    each (query, document) pair jointly through a transformer.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Loaded cross-encoder reranker: {model_name}")

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank search results using cross-encoder scores.

        Args:
            query: The user's search query.
            results: List of results from bi-encoder retrieval.
                     Each result must have a 'metadata' dict with product fields.
            top_k: Number of top results to return after reranking.

        Returns:
            Top-k results re-sorted by cross-encoder relevance score.
        """
        if not results:
            return []

        # Build document texts from product metadata for cross-encoder scoring
        pairs = []
        for r in results:
            m = r["metadata"]
            doc_text = self._build_document_text(m)
            pairs.append((query, doc_text))

        # Score all (query, document) pairs
        scores = self.model.predict(pairs)

        # Attach cross-encoder scores to results
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])

        # Sort by cross-encoder score (higher = more relevant)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    def _build_document_text(self, metadata: Dict) -> str:
        """Build a text representation of the product for cross-encoder input."""
        parts = []

        name = metadata.get("name", "")
        if name:
            parts.append(name)

        brand = metadata.get("brandName", "")
        if brand:
            parts.append(f"Brand: {brand}")

        categories = metadata.get("categoryName", [])
        if categories:
            parts.append(f"Categories: {', '.join(categories)}")

        gender = metadata.get("gender", [])
        if gender:
            parts.append(f"Gender: {', '.join(gender)}")

        price = metadata.get("sellingPrice", 0)
        if price:
            parts.append(f"Price: {price} KWD")

        return " | ".join(parts)


# Quick test
if __name__ == "__main__":
    reranker = Reranker()

    # Simulated results from bi-encoder
    mock_results = [
        {"metadata": {"name": "Vanilla Body Spray 100ml", "brandName": "Adore",
                      "categoryName": ["Body Spray"], "gender": ["Female"], "sellingPrice": 5.0}},
        {"metadata": {"name": "24-Hour Vanilla EDP for Her", "brandName": "OSMA",
                      "categoryName": ["Eau de Parfum"], "gender": ["Female"], "sellingPrice": 15.0}},
        {"metadata": {"name": "Vanilla Licious Body Mist", "brandName": "Adore",
                      "categoryName": ["Body Mist"], "gender": ["Female"], "sellingPrice": 10.0}},
    ]

    query = "long lasting vanilla perfume for women"
    reranked = reranker.rerank(query, mock_results, top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nReranked results:")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. {r['metadata']['name']} (score: {r['rerank_score']:.4f})")
