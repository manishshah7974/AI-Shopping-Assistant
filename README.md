# AI Shopping Assistant — Fragrance & Beauty

A RAG-powered (Retrieval-Augmented Generation) shopping assistant for a Kuwait-based beauty & fragrance e-commerce store. It uses semantic search over 20,000+ products and an LLM for conversational recommendations.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Embedding    │────▶│ FAISS Index  │────▶│ LLM (Groq)   │
│ (MiniLM-L6) │     │ (20k vectors)│     │ (Llama 3.3)  │
└──────────────┘     └──────────────┘     └──────────────┘
    384-dim vectors    Similarity search    Conversational response
```

All 6 use cases share the same core pipeline — the difference is only in what prompt goes to the LLM and whether metadata filtering is applied after retrieval.

---

## Project Structure

```
AI Shopping Assistant Project/
├── .env                              ← Groq API key
├── main.py                           ← Interactive CLI entry point
├── search.py                         ← ShoppingAssistant class (all 6 use cases)
├── vectorstore.py                    ← FAISS index build/load/search
├── embedding.py                      ← Sentence-transformer embedding model
├── data_loader.py                    ← JSON loader + text/metadata structuring
├── clean_json.py                     ← One-time data cleaning script
├── Fragrance_and_Beauty_cleaned.json ← Cleaned product dataset (20k products)
└── faiss_index/                      ← Pre-built FAISS index
    ├── index.faiss
    └── metadata.pkl
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install sentence-transformers faiss-cpu langchain-groq python-dotenv redisvl redis
py -m pip install sentence-transformers faiss-cpu langchain-groq python-dotenv redisvl redis [For Windows]
```

### API Key

Get a free key from [Groq Console](https://console.groq.com/keys) and add it to `.env`:

```
GROQ_API_KEY=your_key_here
```

> The assistant works **without** an API key for search/filter/similar features. Only `/chat` and `/compare` require the LLM.

### Run

```bash
python main.py
```

On first run, the FAISS index is built automatically (~5 minutes for 20k products). Subsequent runs load instantly from disk.

---

## Commands

| Command | Use Case | Needs API Key? |
|---------|----------|----------------|
| `/search vanilla perfume for women` | Natural language search | No |
| `/similar Vanilla S Eau de Parfum - 75ml` | Find similar products | No |
| `/chat I need a gift for my mom, floral, budget 30` | Conversational assistant | Yes |
| `/compare Calvin Klein vs OSMA` | Compare products/brands | Yes |
| `/filter skincare` (then follow prompts) | Smart filtering | No |
| Just type anything | Auto: chat if API key set, else search | Depends |
| `/quit` | Exit | — |

---

## The 6 Use Cases

### 1. Natural Language Product Search (Easiest — minimal changes)

> "Show me vanilla-based perfumes under 20 dinars for women"

The RAG pipeline embeds product descriptions and the user query into the same 384-dimensional vector space. FAISS finds the most semantically similar products. Much better than keyword search because it understands intent — "long-lasting scent" matches products mentioning "24-hour wear" even without exact word overlap.

**How it works in code:**
- `search.py` → `search_products(query, top_k)` calls `VectorStore.search()`
- `vectorstore.py` → embeds the query using `EmbeddingModel.embed_query()`, runs `faiss.IndexFlatL2.search()`, returns top-K results with metadata

**Example:**

    You: /search vanilla perfume for women

      1. Vanilla S Eau de Parfum - 75ml
         Brand: OSMA | Price: 5.69 | In Stock
      2. Nude Vanilla Hair & Body Perfume Mist - 236ml
         Brand: Calvin Klein | Price: 13.75 | In Stock
      3. Vanilla Licious All Over Body Spray - 100ml
         Brand: Adore | Price: 10.4 | In Stock

---

### 2. Product Recommendation Engine

> "I like oud and musk fragrances, what else would I enjoy?"

Uses vector similarity to find products with similar scent profiles/ingredients. The embeddings naturally cluster products with similar descriptions together — oud fragrances end up near other oud fragrances in the 384-dimensional space.

**How it works in code:**
- `search.py` → `recommend_similar(product_name)` finds the product in the index, then calls `VectorStore.search_by_index()`
- `vectorstore.py` → `search_by_index(idx)` reconstructs the product's vector using `faiss.reconstruct()` and finds its nearest neighbors

**Example:**

    You: /similar Vanilla S Eau de Parfum - 75ml

      1. N6 Eau de Parfum - 75ml
         Brand: IMAA | Price: 15.659
      2. Amber S Eau de Parfum - 20ml
         Brand: OSMA | Price: 3.24

---

### 3. Conversational Shopping Assistant (Best showcase feature)

> "I need a gift for my mom, she likes floral scents, budget is 30-50"

Full RAG pipeline: retrieves relevant products via semantic search, then feeds them as context to the LLM which acts as a shopping advisor — filtering by constraints (gender, price, scent) and presenting options conversationally.

**How it works in code:**
- `search.py` → `chat(query)` retrieves top-K products, formats them into a context string with name/brand/price/gender
- Sends a shopping-assistant system prompt + product context to Groq's Llama 3.3 70B
- The LLM sees real product data and generates a natural response referencing **only actual products** from the store

**Example:**

    You: /chat is there any perfume which lasts 24 hours?

    Assistant: We have a few long-lasting perfumes that might interest you.
    I'd recommend the Forever Light Fragrance by Ateej, currently on sale
    for 12 KWD (20% off). Another option is the Freshness Eau de Parfum
    by TWO MARK, priced at 21.13 KWD for 75ml.

---

### 4. Compare & Contrast Products

> "What's the difference between Brand X and Brand Y perfumes?"

Retrieves products from both brands via semantic search, feeds both sets to the LLM, and gets a structured comparison covering price range, scent families, and target audience.

**How it works in code:**
- `search.py` → `compare(product_a, product_b)` retrieves top-3 products for each, formats both groups, sends to LLM with a comparison prompt
- The LLM highlights differences in price, scent notes, brand positioning, and target audience

**Example:**

    You: /compare Calvin Klein vs OSMA

---

### 5. Smart Filtering with Natural Language

> "Best deals on skincare right now" or "What's in stock for men under 10?"

Combines vector search with metadata filtering. First retrieves semantically relevant candidates via embeddings, then post-filters on structured fields (price, discount, inventory, gender).

**How it works in code:**
- `search.py` → `filtered_search(query, max_price, gender, category, in_stock)` retrieves top-20 via FAISS, then filters by metadata constraints
- Supports filtering by: max price, gender, category, and stock availability

**Example:**

    You: /filter skincare
      Max price (or Enter to skip): 15
      Gender [Male/Female] (or Enter to skip): Female

      Found 19 products under 15 for women
      1. Comfort 24H Fluid - 40ml
         Brand: Avene | Price: 7.63 | In Stock
      2. Shea Butter Body Lotion Dry Skin - 500ml
         Brand: L'Occitane | Price: 8.4 | In Stock

---

### 6. "Find Similar" Feature

> User is viewing Product X → "Show me similar products"

Embeds the current product, queries FAISS for nearest neighbors. Pure vector similarity — no LLM needed.

**How it works in code:**
- `search.py` → `find_similar(product_name)` looks up the product's index in metadata, then calls `VectorStore.search_by_index()`
- `vectorstore.py` → reconstructs the product's embedding vector and finds the closest vectors in the index (excluding itself)

**Example:**

    You: /similar Vanilla S Eau de Parfum - 75ml

      1. N6 Eau de Parfum - 75ml
         Brand: IMAA | Price: 15.659
      2. Memories Collection Eau de Parfum Set - 4 pcs
         Brand: K.Arthur | Price: 0.25
      3. Amber S Eau de Parfum - 20ml
         Brand: OSMA | Price: 3.24

---

## How the Code Works (File by File)

### `data_loader.py` — Data Ingestion

Converts raw JSON into two things per product:

1. **`text`** — A human-readable string combining all fields. This is what the embedding model reads to understand the product's meaning:

       Product: Nibras Roshoosh - 200ml
       Brand: The Real Fouz
       Description: Key Notes: Coconut, Elemi...
       Categories: Arabic Fragrances, Women...
       Gender: Female, Male
       Price: 15
       Type: Fragrance

2. **`metadata`** — Structured fields (price, brand, gender, inventory, discount) kept separate for filtering at query time. This is how "under 20 dinars" or "for women" filtering works without relying on the LLM.

### `embedding.py` — Text to Vectors

Converts product text into 384-dimensional numerical vectors using `all-MiniLM-L6-v2` (a sentence-transformer model).

- `embed_products(documents)` — Encodes all 20,000 products into vectors. Shape: `(20000, 384)`
- `embed_query(query)` — Encodes a single user query into the same vector space

When a user asks "vanilla perfume for women", the query vector will be mathematically close to product vectors that mention vanilla, feminine fragrances, etc. — even if the exact words don't match.

### `vectorstore.py` — FAISS Index & Search

Stores all 20,000 product embeddings in a FAISS index for fast similarity search.

- `build_index(json_path)` — One-time: loads products → embeds all 20k → stores in FAISS + saves to disk
- `load()` — Loads the pre-built index from `faiss_index/` (instant)
- `search(query, top_k)` — Embeds query → finds top-K nearest products → returns metadata
- `search_by_index(product_idx, top_k)` — Given a product index, finds its most similar products (for "Find Similar")

### `search.py` — ShoppingAssistant Class (All 6 Use Cases)

The brain of the application. Contains all use case methods:

| Method | Use Case | Needs LLM? |
|--------|----------|------------|
| `search_products(query)` | Semantic search → returns products | No |
| `recommend_similar(product_name)` | Finds product, returns nearest neighbors | No |
| `chat(query)` | Retrieves products + LLM generates conversational recommendation | Yes |
| `compare(product_a, product_b)` | Retrieves both, LLM compares them | Yes |
| `filtered_search(query, max_price, gender, ...)` | Semantic search + post-filter on metadata | No |
| `find_similar(product_name)` | Pure vector similarity from a product's embedding | No |

### `main.py` — CLI Entry Point

Interactive command-line interface that routes user input to the appropriate use case method. Handles index building on first run and provides a clean command interface.

### `clean_json.py` — One-Time Data Cleaning

Transforms the raw Elasticsearch export into a clean JSON format:
- Extracts fields from `_source` and `fields`
- Maps gender codes (`4194` → Female, `2741` → Male)
- Strips HTML from descriptions
- Outputs `Fragrance_and_Beauty_cleaned.json`

---

## Why RAG Works Here

The same pipeline powers all 6 features because:

| Use Case | Retrieval | Generation |
|----------|-----------|------------|
| 1. Natural language search | Query → FAISS → top-K products | None (just display) |
| 2. Recommendations | Product description → FAISS → similar products | None |
| 3. Conversational assistant | Query → FAISS → relevant products | LLM filters by constraints & converses |
| 4. Compare products | Brand/product names → FAISS → both sets | LLM compares |
| 5. Smart filtering | Query → FAISS → candidates, then filter metadata | None |
| 6. Find similar | Product embedding → FAISS → nearest neighbors | None |

The intelligence comes from:
- **Rich text embeddings** — The combined text (name + brand + description + categories + gender + price) ensures semantic search finds products by scent notes, brand, category, or description
- **Structured metadata** — Enables post-retrieval filtering (price range, gender, in-stock, discounts) without relying on the LLM
- **Shopping-assistant prompt** — Tells the LLM to act as a Kuwait-based beauty advisor, use KWD currency, and only recommend actual products from the retrieved context

---

## Sample Output

    ============================================================
      AI Shopping Assistant - Fragrance & Beauty
    ============================================================

    Commands:
      /search <query>        - Search products
      /chat <query>          - Chat with AI assistant (needs API key)
      /compare <a> vs <b>    - Compare products/brands (needs API key)
      /similar <product>     - Find similar products
      /filter <query>        - Filtered search (follow prompts)
      /quit                  - Exit

Query 1:
You: /search sandalwood fragnance perfumes 

  1. Flowers W Eau de Parfum - 75ml
     Brand: OSMA | Price: 5.69 (70% off) | In Stock
     Gender: Female, Male
     SKU: ORL-00007219
     URL: https://www.boutiqaat.com/en-kw/women/osma-perfume-w-flowers-75ml-orl-00007219-1/p/
  2. Mood 11 Home Scent Spray - 200ml
     Brand: Reef Perfumes | Price: 5.8 (60% off) | In Stock
     Gender: Female, Male
     SKU: ORL-00004155
     URL: https://www.boutiqaat.com/en-kw/women/mood-11-home-scent-spray-200ml-unisex-by-reef-perfumes-orl-00004155/p/
  3. More S Eau de Parfum - 100ml
     Brand: Charlotte Tilbury | Price: 56 | In Stock
     Gender: Female, Male
     SKU: FR-00005989
     URL: https://www.boutiqaat.com/en-kw/women/charlotte-tilbury-fragrance-100ml-more-s-me-fr-00005989-1/p/
  4. Flower Lust Extrait de Parfum - 90ml
     Brand: d'Orsay | Price: 89 | In Stock
     Gender: Female, Male
     SKU: FR-00006727
     URL: https://www.boutiqaat.com/en-kw/women/dorsay-flower-lust-extdeparfum-90ml-fr-00006727-1/p/
  5. More S Eau de Parfum - 10ml
     Brand: Charlotte Tilbury | Price: 10 | In Stock
     Gender: Female, Male
     SKU: FR-00005995
     URL: https://www.boutiqaat.com/en-kw/women/charlotte-tilbury-fragrance-10ml-more-s-me-fr-00005995-1/p/

Query 2:
You: /chat  best honey aud perfume between 30 kwd to 50 kwd top 5

Assistant: Based on your query, I'd recommend the following top 5 honey/oud perfumes between 30 KWD to 50 KWD:

1. Oud For Greatness Hair Perfume (35.25 KWD) by Initio - https://www.boutiqaat.com/en-kw/women/initio-oud-for-greatness-hair-perfume-50ml-fr-00004782-1/p/
2. Unfortunately, we don't have many options in this range, but I can suggest the Honeysuckle & Davana Cologne (23.75 KWD) by Jo Malone London, which is slightly below your budget - https://www.boutiqaat.com/en-kw/women/jo-malice-london-honeysuckle-davana-cologne-fr-00003054/p/        
3. Since there aren't many options that fit your exact criteria, I'd like to suggest the Bullet Eau de Parfum (24.5 KWD) by Arabian Oud, which is also slightly below your budget - https://www.boutiqaat.com/en-kw/women/bullet-perfume-100ml-orl-00009080-1/p/
4. The Small Zayed Garden Candle (34.6 KWD) by Alchimie is another option, although it's a candle, not a perfume - https://www.boutiqaat.com/en-kw/women/alchimie-zayed-garden-candle-olivee-medium-ac-00016095-1/p/
5. The Cologne Discovery Collection (6.5 KWD) by Jo Malone London is not within your budget range, but it's a great way to try out different scents - https://www.boutiqaat.com/en-kw/women/jo-malone-london-cologne-discovery-collection-fr-00003095/p/

Please note that options are limited, and most of these products are either slightly below your budget or not exactly what you're looking for. I hope this helps, and I'm happy to assist you further!