"""Small local product catalog used when an API request provides product_id only."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "products" / "sample_products.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict]:
    if not CATALOG_PATH.exists():
        return {}
    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {record["product_id"]: record for record in records if record.get("product_id")}


def get_product(product_id: str | None) -> dict | None:
    return load_catalog().get(product_id) if product_id else None
