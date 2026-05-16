import json
from typing import List, Dict


def load_products(json_path: str) -> List[Dict]:
    """
    Load products from JSON and structure them for RAG.
    Returns list of dicts with 'text' (for embedding) and 'metadata' (for filtering).
    """
    with open(json_path, encoding="utf-8") as f:
        products = json.load(f)

    documents = []
    for product in products:
        # Combine fields into a single searchable text
        text = build_product_text(product)

        # Keep structured metadata for filtering
        metadata = {
            "name": product.get("name", ""),
            "brandName": product.get("brandName", ""),
            "price": product.get("price", 0),
            "sellingPrice": product.get("sellingPrice", 0),
            "discountPercentage": product.get("discountPercentage", 0),
            "gender": product.get("gender", []),
            "inventory": product.get("inventory", 0),
            "categoryName": product.get("categoryName", []),
            "attributeSetName": product.get("attributeSetName", ""),
            "slug": product.get("slug", ""),
        }

        documents.append({"text": text, "metadata": metadata})

    print(f"[INFO] Loaded {len(documents)} products from {json_path}")
    return documents


def build_product_text(product: Dict) -> str:
    """
    Build a rich text representation of a product for embedding.
    Combines name, brand, description, categories, gender, and price info.
    """
    parts = []

    name = product.get("name", "")
    if name:
        parts.append(f"Product: {name}")

    brand = product.get("brandName", "")
    if brand:
        parts.append(f"Brand: {brand}")

    desc = product.get("description", "")
    if desc:
        parts.append(f"Description: {desc}")

    categories = product.get("categoryName", [])
    if categories:
        # Deduplicate categories
        unique_cats = list(dict.fromkeys(categories))
        parts.append(f"Categories: {', '.join(unique_cats)}")

    gender = product.get("gender", [])
    if gender:
        parts.append(f"Gender: {', '.join(gender)}")

    price = product.get("sellingPrice", 0)
    discount = product.get("discountPercentage", 0)
    if price:
        price_str = f"Price: {price}"
        if discount > 0:
            price_str += f" ({discount}% off)"
        parts.append(price_str)

    attr_set = product.get("attributeSetName", "")
    if attr_set:
        parts.append(f"Type: {attr_set.replace('_set', '')}")

    return "\n".join(parts)


# Quick test
if __name__ == "__main__":
    docs = load_products("Fragrance_and_Beauty_cleaned.json")
    print(f"\n--- Example product text ---\n{docs[0]['text']}")
    print(f"\n--- Example metadata ---\n{docs[0]['metadata']}")
