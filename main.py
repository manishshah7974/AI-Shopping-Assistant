"""AI Shopping Assistant - Main Entry Point"""
import os
from vectorstore import VectorStore
from search import ShoppingAssistant

JSON_PATH = "Fragrance_and_Beauty_cleaned.json"
INDEX_DIR = "faiss_index"


def build_index():
    """Build FAISS index from product data (one-time setup)."""
    store = VectorStore(INDEX_DIR)
    store.build_index(JSON_PATH)
    print("\nIndex built! You can now run queries.")


def main():
    """Interactive shopping assistant."""
    # Build index if not exists
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("No index found. Building index first (this takes ~5 minutes)...")
        build_index()

    assistant = ShoppingAssistant(INDEX_DIR)

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
            # Default: treat as chat if API key exists, else search
            if os.getenv("GROQ_API_KEY"):
                response = assistant.chat(user_input, top_k=5)
                print(f"\nAssistant: {response}\n")
            else:
                results = assistant.search_products(user_input, top_k=5)
                print_results(results)


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
        print(f"  {i}. {m['name']}")
        print(f"     Brand: {m['brandName']} | Price: {m['sellingPrice']}{discount} | {stock}")
        print(f"     Gender: {', '.join(m['gender'])}")
    print()


if __name__ == "__main__":
    main()
