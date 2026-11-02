"""Deterministic product-query parsing used by the graph's first node."""

from __future__ import annotations

import re

from .catalog import get_product
from .llm import LocalLLM
from .schemas import AssessmentRequest, ProductSummary


BRANDS = {
    "Levi's": (r"\blevi(?:'s|s| strauss)?\b",),
    "Patagonia": (r"\bpatagonia\b",),
    "H&M": (r"\bh\s*(?:&|and)\s*m\b", r"\bhm\b"),
}

CATEGORIES = {
    "jeans": ("jean", "denim pants", "denim trouser"),
    "t-shirt": ("t-shirt", "t shirt", "tee"),
    "jacket": ("jacket", "puffer", "parka", "coat"),
    "shirt": ("shirt", "blouse"),
    "trousers": ("trouser", "pants"),
}

MATERIALS = {
    "organic cotton": ("organic cotton",),
    "recycled cotton": ("recycled cotton",),
    "cotton": ("cotton",),
    "denim": ("denim",),
    "recycled polyester": ("recycled polyester", "rpet"),
    "polyester": ("polyester",),
    "nylon": ("nylon",),
    "wool": ("wool",),
}

SUSTAINABILITY_KEYWORDS = (
    "organic",
    "recycled",
    "certified",
    "gots",
    "fair trade",
    "bluesign",
    "circular",
    "waterless",
    "sustainable",
)


def canonical_brand(value: str | None) -> str | None:
    if not value:
        return None
    for brand, patterns in BRANDS.items():
        if any(re.search(pattern, value, flags=re.I) for pattern in patterns):
            return brand
    return value.strip()


def _first_match(text: str, mapping: dict[str, tuple[str, ...]]) -> str | None:
    lowered = text.lower()
    for canonical, aliases in mapping.items():
        if any(alias in lowered for alias in aliases):
            return canonical
    return None


def parse_product(request: AssessmentRequest, llm: LocalLLM | None = None) -> ProductSummary:
    catalog_product = get_product(request.product_id)
    text = request.query.strip() or (catalog_product or {}).get("query", "")
    brand = canonical_brand(request.brand or (catalog_product or {}).get("brand"))
    if not brand:
        for candidate, patterns in BRANDS.items():
            if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
                brand = candidate
                break

    catalog_category = (catalog_product or {}).get("category")
    category = request.category.strip().lower() if request.category else (catalog_category or _first_match(text, CATEGORIES))

    detected: list[str] = []
    source_materials = request.materials or (catalog_product or {}).get("materials", [])
    supplied = [material.strip().lower() for material in source_materials if material.strip()]
    for material in supplied:
        if material not in detected:
            detected.append(material)
    lowered = text.lower()
    for canonical, aliases in MATERIALS.items():
        if any(alias in lowered for alias in aliases) and canonical not in detected:
            detected.append(canonical)
    # Denim is normally cotton-based; retaining both improves material retrieval while
    # leaving the final assessment explicit about the inference.
    if "denim" in detected and not any("cotton" in item for item in detected):
        detected.append("cotton")

    keywords = [keyword for keyword in SUSTAINABILITY_KEYWORDS if keyword in lowered]
    used_llm = False
    # The model is a conservative fallback: it may fill genuinely missing
    # attributes from a controlled vocabulary, but never overwrite supplied or
    # deterministic values and never invent a free-form value.
    if llm and llm.enabled and (not brand or not category or not detected):
        hints = llm.extract_product_hints(
            text,
            allowed_brands=list(BRANDS),
            allowed_categories=list(CATEGORIES),
            allowed_materials=list(MATERIALS),
            allowed_keywords=list(SUSTAINABILITY_KEYWORDS),
        )
        if hints:
            if not brand and hints.brand:
                brand = hints.brand
                used_llm = True
            if not category and hints.category:
                category = hints.category
                used_llm = True
            for material in hints.materials:
                if material not in detected:
                    detected.append(material)
                    used_llm = True
            for keyword in hints.sustainability_keywords:
                if keyword not in keywords:
                    keywords.append(keyword)
    missing: list[str] = []
    if not brand:
        missing.append("brand")
    if not category:
        missing.append("category")
    if not detected:
        missing.append("materials")

    return ProductSummary(
        product_id=request.product_id,
        query=text,
        brand=brand,
        category=category,
        materials=detected,
        sustainability_keywords=keywords,
        missing_attributes=missing,
        extraction_method="rules_and_llm_fallback" if used_llm else "rules",
    )
