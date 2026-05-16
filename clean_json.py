import json
import re

INPUT = r"C:\Users\24IN214\Desktop\AI Shopping Assistant Project\AI Shopping Assistant\Fragrance_and_Beauty_dataset_.json"
OUTPUT = r"C:\Users\24IN214\Desktop\AI Shopping Assistant Project\AI Shopping Assistant\Fragrance_and_Beauty_cleaned.json"

GENDER_MAP = {"4194": "Female", "2741": "Male"}

def clean_html(text):
    return re.sub(r'<[^>]+>', '', text) if text else ""

def transform(item):
    src = item.get("_source", {})
    fields = item.get("fields", {})

    return {
        "attributeSetName": src.get("attributeSetName"),
        "gender": [GENDER_MAP.get(g, g) for g in src.get("gender", [])],
        "description": clean_html(src.get("enDescription", "")),
        "name": src.get("enName"),
        "sku": src.get("sku"),
        "inventory": src.get("inventory"),
        "brandName": src.get("brand", {}).get("englishName"),
        "brandId": src.get("brand", {}).get("id"),
        "slug": src.get("slug"),
        "categoryName": src.get("enCategoryName", []),
        "sellingPrice": fields.get("selling_price", [None])[0],
        "discountPercentage": fields.get("discount", [None])[0],
        "price": fields.get("price", [None])[0],
    }

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = [transform(item) for item in data]

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=4, ensure_ascii=False)

print(f"Done! Cleaned {len(cleaned)} items -> {OUTPUT}")
