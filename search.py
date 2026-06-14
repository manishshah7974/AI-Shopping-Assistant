import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from vectorstore import VectorStore

load_dotenv()

# ─── Redis Semantic Cache Setup ───
REDIS_URL = os.getenv("REDIS_URL", "")
semantic_cache = None

try:
    from redisvl.extensions.cache.llm import SemanticCache

    semantic_cache = SemanticCache(
        name="shopping_llm_cache",
        redis_url=REDIS_URL,
        distance_threshold=0.2,
        ttl=86400,
    )
    print("[INFO] Redis semantic cache enabled.")
except Exception as e:
    print(f"[WARN] Redis semantic cache not available: {e}")


class ShoppingAssistant:
    def __init__(self):
        self.store = VectorStore()
        self.store.load()

        # LLM setup (Groq)
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
        print("[INFO] Shopping Assistant ready.")

    # ─── USE CASE 1: Natural Language Search ───
    def search_products(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search - returns matching products."""
        return self.store.search(query, top_k=top_k)

    # ─── USE CASE 2: Product Recommendations ───
    def recommend_similar(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """Find products similar to a given product name."""
        return self._similar_via_redis(product_name, top_k)

    # ─── USE CASE 3: Conversational Shopping Assistant ───
    def chat(self, query: str, top_k: int = 5) -> str:
        """Full RAG: retrieve products + LLM generates conversational response."""
        results = self.store.search(query, top_k=top_k)
        context = self._format_products(results)

        prompt = f"""You are a helpful shopping assistant for a beauty & fragrance e-commerce store based in Kuwait.
All prices are in KWD (Kuwaiti Dinar). Always display prices as "X KWD", never use $ or dollars.
Based on the customer's query and the available products below, provide a helpful recommendation.
Be concise, friendly, and mention specific product names, prices, and brands.
Always include the product URL so the customer can view or purchase it.
Only recommend products from the list below - do not make up products.

Customer query: "{query}"

Available products:
{context}

Response:"""
        return self._cached_llm_call(prompt, cache_key=query)

    # ─── USE CASE 4: Compare Products ───
    def compare(self, product_a: str, product_b: str) -> str:
        """Compare two products or brands using RAG."""
        results_a = self.store.search(product_a, top_k=3)
        results_b = self.store.search(product_b, top_k=3)

        context_a = self._format_products(results_a)
        context_b = self._format_products(results_b)

        prompt = f"""You are a shopping assistant for a Kuwait-based beauty & fragrance store.
All prices are in KWD (Kuwaiti Dinar). Compare these two product groups concisely.
Highlight differences in price, scent notes, brand, and target audience.
Include product URLs for each product mentioned.

Group A - "{product_a}":
{context_a}

Group B - "{product_b}":
{context_b}

Comparison:"""
        return self._cached_llm_call(prompt, cache_key=f"{product_a} vs {product_b}")

    # ─── USE CASE 5: Smart Filtering (search + metadata filter) ───
    def filtered_search(self, query: str, top_k: int = 20,
                        max_price: Optional[float] = None,
                        gender: Optional[str] = None,
                        category: Optional[str] = None,
                        in_stock: bool = False) -> List[Dict]:
        """Semantic search with post-retrieval metadata filtering."""
        results = self.store.search(query, top_k=top_k)
        filtered = []
        for r in results:
            m = r["metadata"]
            if max_price and m["sellingPrice"] > max_price:
                continue
            if gender and gender.capitalize() not in m["gender"]:
                continue
            if in_stock and m["inventory"] <= 0:
                continue
            if category and not any(category.lower() in c.lower() for c in m["categoryName"]):
                continue
            filtered.append(r)
        return filtered

    # ─── USE CASE 6: Find Similar Products ───
    def find_similar(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """Given a product name, find the most similar products."""
        return self._similar_via_redis(product_name, top_k)

    # ─── Helpers ───
    def _similar_via_redis(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """Semantic search to find the product, then use its vector for similarity."""
        # Step 1: Find the product in Redis via semantic search
        matches = self.store.search(product_name, top_k=1)
        if not matches:
            return []
        matched = matches[0]
        # Step 2: Fetch that product's actual stored embedding from Redis
        embedding = self.store.get_embedding_by_idx(matched["product_idx"])
        if embedding is not None:
            return self.store.search_by_vector(
                embedding.tolist(), top_k=top_k,
                exclude_name=matched["metadata"]["name"]
            )
        # Fallback: use text search results
        return self.store.search(product_name, top_k=top_k + 1)[1:]

    def _cached_llm_call(self, prompt: str, cache_key: str) -> str:
        """Call LLM with Redis semantic cache. cache_key is the user query only."""
        if semantic_cache:
            cached = semantic_cache.check(prompt=cache_key)
            if cached:
                print("  [CACHE HIT]")
                return cached[0]["response"]
        response = self.llm.invoke([prompt])
        result = response.content
        if semantic_cache:
            semantic_cache.store(prompt=cache_key, response=result)
        return result

    def _format_products(self, results: List[Dict]) -> str:
        """Format product results into readable text for LLM context."""
        lines = []
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            discount = f" ({m['discountPercentage']}% off)" if m["discountPercentage"] > 0 else ""
            url = f"https://www.boutiqaat.com/en-kw/women/{m['slug']}" if m.get("slug") else ""
            lines.append(f"{i}. {m['name']} | Brand: {m['brandName']} | "
                         f"Price: {m['sellingPrice']} KWD{discount} | Gender: {', '.join(m['gender'])} | "
                         f"SKU: {m.get('sku', '')} | URL: {url}")
        return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    assistant = ShoppingAssistant()

    # Test Use Case 1: Search
    print("\n═══ USE CASE 1: Natural Language Search ═══")
    results = assistant.search_products("oud perfume for men", top_k=3)
    for r in results:
        m = r["metadata"]
        print(f"  {m['name']} | {m['brandName']} | {m['sellingPrice']}")

    # Test Use Case 5: Filtered Search
    print("\n═══ USE CASE 5: Smart Filtering ═══")
    results = assistant.filtered_search("moisturizer", max_price=15, gender="Female")
    print(f"  Found {len(results)} products under 15 for women")
    for r in results[:3]:
        m = r["metadata"]
        print(f"  {m['name']} | {m['sellingPrice']}")

    # Test Use Case 6: Find Similar
    print("\n═══ USE CASE 6: Find Similar ═══")
    results = assistant.find_similar("Vanilla S Eau de Parfum - 75ml", top_k=3)
    for r in results:
        m = r["metadata"]
        print(f"  {m['name']} | {m['brandName']} | {m['sellingPrice']}")
