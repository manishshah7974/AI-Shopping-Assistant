"""AI Shopping Assistant - Main Entry Point"""
from vectorstore import VectorStore
from search import ShoppingAssistant

JSON_PATH = "Fragrance_and_Beauty_cleaned.json"


def build_index():
    """Build Redis vector index from product data (one-time setup)."""
    store = VectorStore()
    store.build_index(JSON_PATH)
    print("\nIndex built! You can now run queries.")


def main():
    """Interactive shopping assistant."""
    # Build index if not exists or is empty in Redis
    store = VectorStore()
    if not store.index.exists():
        print("No Redis index found. Building index first (this takes a few minutes)...")
        store.build_index(JSON_PATH)
    else:
        info = store.index.info()
        num_docs = int(info.get("num_docs", 0))
        if num_docs == 0:
            print("[WARN] Index exists but is empty. Rebuilding (this takes a few minutes)...")
            store.build_index(JSON_PATH)

    assistant = ShoppingAssistant()

    print("\n" + "=" * 60)
    print("  AI Shopping Assistant - Fragrance & Beauty")
    print("=" * 60)
    print("\nCommands:")
    print("  /search <query>        - Search products")
    print("  /chat <query>          - Chat with AI assistant (needs API key)")
    print("  /compare <a> vs <b>    - Compare products/brands (needs API key)")
    print("  /similar <product>     - Find similar products")
    print("  /filter <query>        - Filtered search (follow prompts)")
    print("  /quit                  - Exit")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        elif user_input.startswith("/search "):
            query = user_input[8:]
            results = assistant.search_products(query, top_k=5)
            print_results(results)

        elif user_input.startswith("/chat "):
            query = user_input[6:]
            response = assistant.chat(query, top_k=5)
            print(f"\nAssistant: {response}\n")

        elif user_input.startswith("/compare "):
            parts = user_input[9:].split(" vs ")
            if len(parts) == 2:
                response = assistant.compare(parts[0].strip(), parts[1].strip())
                print(f"\n{response}\n")
            else:
                print("Usage: /compare <product A> vs <product B>")

        elif user_input.startswith("/similar "):
            product = user_input[9:]
            results = assistant.find_similar(product, top_k=5)
            print_results(results)

        elif user_input.startswith("/filter "):
            query = user_input[8:]
            max_price = input("  Max price (or Enter to skip): ").strip()
            gender = input("  Gender [Male/Female] (or Enter to skip): ").strip()
            results = assistant.filtered_search(
                query,
                max_price=float(max_price) if max_price else None,
                gender=gender if gender else None,
            )
            print_results(results[:5])

        else:
            # Default: always use chat mode
            response = assistant.chat(user_input, top_k=5)
            print(f"\nAssistant: {response}\n")


def print_results(results):
    """Display product results."""
    if not results:
        print("\n  No results found.\n")
        return
    print()
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        discount = f" ({m['discountPercentage']}% off)" if m["discountPercentage"] > 0 else ""
        stock = "In Stock" if m["inventory"] > 0 else "Out of Stock"
        url = f"https://www.boutiqaat.com/en-kw/women/{m['slug']}" if m.get("slug") else ""
        print(f"  {i}. {m['name']}")
        print(f"     Brand: {m['brandName']} | Price: {m['sellingPrice']}{discount} | {stock}")
        print(f"     Gender: {', '.join(m['gender'])}")
        if m.get("sku"):
            print(f"     SKU: {m['sku']}")
        if url:
            print(f"     URL: {url}")
    print()


if __name__ == "__main__":
    main()
