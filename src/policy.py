"""Transparent policies for coverage, conflicts, scores, and confidence.

The rules are deliberately small and inspectable: they describe how this
assignment prototype behaves rather than pretending that a heuristic is a
lifecycle-assessment model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean
from typing import Iterable

from .schemas import Contradiction, DimensionScore, ProductSummary


@dataclass(frozen=True)
class DepthPolicy:
    top_k: int
    max_rounds: int
    sources: tuple[str, ...]
    check_conflicts: bool


DEPTH_POLICIES = {
    "quick": DepthPolicy(3, 1, ("brand_evidence", "material_evidence"), True),
    "standard": DepthPolicy(5, 2, ("brand_evidence", "material_evidence", "public_signal"), True),
    "deep": DepthPolicy(
        8,
        3,
        ("brand_evidence", "material_evidence", "certification_evidence", "public_signal"),
        True,
    ),
}

AUTHORITY = {"primary": 1.0, "secondary": 0.72, "public_signal": 0.42, "unknown": 0.3}
POSITIVE_TERMS = {
    "progress", "reduced", "reduction", "renewable", "certified", "circular", "recycled",
    "responsible", "target", "achieved", "organic", "repair", "reuse", "traceability",
}
NEGATIVE_TERMS = {
    "greenwashing", "misleading", "criticism", "concern", "failed", "failure", "pollution",
    "poor", "not transparent", "overproduction", "waste", "lawsuit", "harm", "unethical",
    "forced labor", "forced labour", "burning", "ripped", "fell apart",
}
DIMENSION_TERMS = {
    "water": {"water", "dyeing", "irrigation"},
    "climate": {"climate", "carbon", "emission", "energy", "renewable"},
    "materials": {"cotton", "polyester", "organic", "recycled", "denim", "material"},
    "certification": {"certif", "gots", "standard", "traceability", "label"},
    "labour_and_supply_chain": {"worker", "labour", "labor", "wage", "forced", "supply chain"},
    "durability": {"durability", "durable", "repair", "ripped", "fell apart", "quality"},
    "waste_and_circularity": {"waste", "overproduction", "landfill", "burning", "circular", "reuse"},
}
CLAIM_TERMS = {
    "reduced", "reduction", "target", "progress", "achieved", "emission", "climate", "water",
    "recycled", "organic", "renewable", "repair", "reuse", "circular", "certified", "traceability",
    "fair trade", "waste", "durability", "supply chain", "worker",
}
BOILERPLATE_SECTION_TERMS = {"purpose", "forward-looking", "remuneration", "b impact assessment", "governance"}


def is_claim_specific(text: str, section: str = "") -> bool:
    """Reject report boilerplate before it can become a product finding."""
    lowered = f"{section} {text}".lower()
    claim_hits = sum(term in lowered for term in CLAIM_TERMS)
    boilerplate = any(term in section.lower() for term in BOILERPLATE_SECTION_TERMS)
    return claim_hits >= (2 if boilerplate else 1)


def best_claim_sentence(text: str) -> str:
    """Select an actual claim sentence, never a document heading as a finding."""
    cleaned = re.sub(r"<[^>]+>|\[([^\]]+)\]\([^)]*\)", r"\1", text)
    cleaned = re.sub(r"\bSection:\s*[^.]{0,220}(?=\.|$)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = [part.strip(" -–—") for part in re.split(r"(?<=[.!?])\s+", cleaned) if len(part.strip()) >= 40]
    for sentence in sentences:
        if is_claim_specific(sentence):
            return sentence[:420]
    return (sentences[0] if sentences else cleaned)[:420]


def evidence_group(item: dict) -> str:
    category = item.get("category")
    if category == "brand":
        return "brand_evidence"
    if category == "certifications":
        return "certification_evidence"
    if category == "materials":
        return "material_evidence"
    if category in {"sentiment", "independent"}:
        return "public_signal"
    return "unknown"


def evidence_quality(required_tools: Iterable[str], evidence: list[dict]) -> dict[str, dict[str, float | int]]:
    """Summarize source quality without treating the number of chunks as proof."""
    quality: dict[str, dict[str, float | int]] = {}
    for tool in required_tools:
        items = [item for item in evidence if evidence_group(item) == tool]
        if not items:
            quality[tool] = {"evidence_count": 0, "source_count": 0, "quality": 0.0}
            continue
        sources = {item.get("source_file") for item in items if item.get("source_file")}
        authority = mean(AUTHORITY.get(item.get("credibility_tier", "unknown"), 0.3) for item in items)
        relevance = mean(float(item.get("retrieval_score", 0.0)) for item in items)
        recency = mean(float(item.get("recency_score", 0.6)) for item in items)
        diversity = min(1.0, len(sources) / 2)
        quality[tool] = {
            "evidence_count": len(items),
            "source_count": len(sources),
            "quality": round(0.40 * authority + 0.25 * relevance + 0.20 * recency + 0.15 * diversity, 4),
        }
    return quality


def sufficiency(required_tools: Iterable[str], evidence: list[dict]) -> tuple[bool, list[str], float]:
    """Check the original source plan, not only the most recent retry plan."""
    required = list(dict.fromkeys(required_tools))
    quality = evidence_quality(required, evidence)
    missing = sorted(tool for tool in required if not quality[tool]["evidence_count"])
    coverage = (len(required) - len(missing)) / len(required) if required else 1.0
    return not missing, missing, round(coverage, 4)


def text_polarity(text: str) -> float:
    lowered = re.sub(r"\s+", " ", text.lower())
    positive = sum(1 for term in POSITIVE_TERMS if term in lowered)
    negative = sum(1 for term in NEGATIVE_TERMS if term in lowered)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def claim_dimensions(text: str) -> set[str]:
    lowered = text.lower()
    return {dimension for dimension, terms in DIMENSION_TERMS.items() if any(term in lowered for term in terms)}


def _dates_comparable(first: dict, second: dict, max_years: int = 3) -> bool:
    """Avoid presenting distant historical claims as a current contradiction."""
    values: list[date] = []
    for item in (first, second):
        value = item.get("published_date")
        if not value:
            return True
        try:
            values.append(datetime.strptime(value, "%Y-%m-%d").date())
        except ValueError:
            return True
    return abs((values[0] - values[1]).days) <= max_years * 365


def detect_conflicts(product: ProductSummary, evidence: list[dict]) -> list[Contradiction]:
    """Flag opposite polarity only where sources address the same claim topic."""
    official = [item for item in evidence if item.get("category") == "brand" and text_polarity(item.get("text", "")) > 0.1]
    external = [
        item for item in evidence
        if item.get("category") in {"independent", "sentiment"} and text_polarity(item.get("text", "")) < -0.1
    ]
    conflicts: list[Contradiction] = []
    seen: set[str] = set()
    for claim in official:
        for signal in external:
            common = claim_dimensions(claim.get("text", "")) & claim_dimensions(signal.get("text", ""))
            if not common or not _dates_comparable(claim, signal):
                continue
            dimension = sorted(common)[0]
            if dimension in seen:
                continue
            seen.add(dimension)
            conflicts.append(
                Contradiction(
                    dimension=dimension,
                    description=(
                        f"{product.brand or 'The brand'} reports a positive {dimension.replace('_', ' ')} claim, "
                        "while an independent or public source raises a negative signal on the same topic. "
                        "The system preserves both sources; this is not a finding that either one is false."
                    ),
                    supporting_evidence=[claim["evidence_id"]],
                    opposing_evidence=[signal["evidence_id"]],
                    severity="medium",
                )
            )
    return conflicts


def _citations(items: list[dict], categories: set[str], limit: int = 4) -> list[str]:
    selected = [item for item in items if item.get("category") in categories]
    selected.sort(key=lambda item: item.get("retrieval_score", 0.0), reverse=True)
    return [item["evidence_id"] for item in selected[:limit]]


DIMENSION_CATEGORIES = {
    "material_sustainability": {"materials", "certifications"},
    "brand_commitments": {"brand", "independent"},
    "certification_and_supply_chain": {"certifications"},
    "public_signal": {"sentiment", "independent"},
}


def validate_dimension_citations(scores: list[dict], evidence: list[dict]) -> tuple[bool, list[str]]:
    """Verify citations exist and belong to the dimension they support."""
    by_id = {item["evidence_id"]: item for item in evidence}
    invalid: list[str] = []
    for score in scores:
        dimension = score.get("dimension")
        expected = DIMENSION_CATEGORIES.get(dimension)
        if score.get("status") == "not_assessable" and score.get("citations"):
            invalid.append(f"{dimension}: citations on a non-assessable dimension")
        if not expected:
            continue
        for citation in score.get("citations", []):
            item = by_id.get(citation)
            if not item or item.get("category") not in expected:
                invalid.append(f"{dimension}: {citation}")
    return not invalid, invalid


def score_assessment(
    product: ProductSummary,
    evidence: list[dict],
    contradictions: list[Contradiction],
    coverage: float,
    unconfirmed_conflicts: int = 0,
) -> tuple[list[DimensionScore], int, float]:
    """Produce inspectable signals; the overall deliberately excludes coverage."""
    materials = " ".join(product.materials).lower()
    material_assessable = bool(product.materials)
    material_score = 0
    material_citations: list[str] = []
    material_rationale = "Material sustainability is not assessable because the product composition is unknown."
    if material_assessable:
        material_score = 52
        if "organic" in materials:
            material_score += 14
        if "recycled" in materials:
            material_score += 12
        if "polyester" in materials and "recycled polyester" not in materials:
            material_score -= 12
        if "denim" in materials:
            material_score -= 3
        material_citations = _citations(evidence, {"materials", "certifications"})
        material_rationale = f"Rule-based material signal for {', '.join(product.materials)}; it is not an LCA."
        if not material_citations:
            material_score = 35
            material_rationale = "Product materials were supplied, but no relevant material evidence was retrieved."

    brand_items = [item for item in evidence if item.get("category") == "brand"]
    brand_sources = {item.get("source_file") for item in brand_items if item.get("source_file")}
    external_corroboration = any(item.get("category") == "independent" for item in evidence)
    brand_assessable = bool(product.brand and brand_items)
    brand_score = 0 if not brand_assessable else min(75, 45 + min(len(brand_sources), 2) * 8 + (4 if external_corroboration else 0))
    if contradictions:
        brand_score -= min(15, 5 * len(contradictions))
    brand_citations = _citations(evidence, {"brand", "independent"})

    certification_items = [item for item in evidence if item.get("category") == "certifications"]
    claimed_certification = any(
        keyword in " ".join(product.sustainability_keywords).lower()
        for keyword in ("certified", "gots", "fair trade", "bluesign")
    )
    certification_assessable = claimed_certification and bool(certification_items)
    certification_score = 0
    certification_rationale = "Certification and supply-chain status is not assessable without product-specific certification evidence."
    if claimed_certification:
        certification_rationale = "A certification was mentioned, but no product-level registry record was available locally."
        certification_assessable = False
    certification_citations = _citations(evidence, {"certifications"})

    public_items = [item for item in evidence if item.get("category") in {"sentiment", "independent"}]
    public_assessable = bool(public_items)
    public_polarity = mean([text_polarity(item["text"]) for item in public_items]) if public_items else 0.0
    sentiment_score = max(20, min(80, round(50 + public_polarity * 25))) if public_assessable else 0
    sentiment_citations = _citations(evidence, {"sentiment", "independent"})

    authority = mean([AUTHORITY.get(item.get("credibility_tier", "unknown"), 0.3) for item in evidence]) if evidence else 0.0
    relevance = mean([item.get("retrieval_score", 0.0) for item in evidence]) if evidence else 0.0
    recency = mean([item.get("recency_score", 0.6) for item in evidence]) if evidence else 0.0
    source_diversity = min(1.0, len({item.get("source_file") for item in evidence}) / 6) if evidence else 0.0
    agreement = 0.72 if contradictions else 1.0
    confidence = 0.32 * coverage + 0.22 * authority + 0.18 * relevance + 0.14 * recency + 0.08 * source_diversity + 0.06 * agreement
    if contradictions:
        confidence -= 0.08
    # Keyword/topic matches that an LLM could not semantically validate are
    # not returned as contradictions, but they still mean the evidence picture
    # is less certain than a clean assessment.
    confidence -= min(0.12, 0.04 * unconfirmed_conflicts)
    if not certification_assessable:
        confidence -= 0.07
    if product.brand and len(brand_sources) < 2:
        confidence -= 0.08
    if "brand" in product.missing_attributes:
        confidence -= 0.20
    if "materials" in product.missing_attributes:
        confidence -= 0.18
    if "category" in product.missing_attributes:
        confidence -= 0.05
    confidence = max(0.05, min(0.95, confidence))
    dimension_confidence = round(confidence, 3)

    scores = [
        DimensionScore(dimension="material_sustainability", score=max(0, min(100, material_score)), status="assessed" if material_assessable else "not_assessable", confidence=dimension_confidence,
                       rationale=material_rationale, citations=material_citations),
        DimensionScore(dimension="brand_commitments", score=max(0, min(100, brand_score)), status="assessed" if brand_assessable else "not_assessable", confidence=dimension_confidence,
                       rationale="Based on distinct official source documents, with a limited adjustment for independent disagreement." if brand_assessable else "Brand commitments are not assessable because no relevant brand evidence was retrieved.", citations=brand_citations),
        DimensionScore(dimension="certification_and_supply_chain", score=certification_score, status="assessed" if certification_assessable else "not_assessable",
                       confidence=round(confidence * (0.8 if not certification_assessable else 1.0), 3), rationale=certification_rationale, citations=certification_citations if certification_assessable else []),
        DimensionScore(dimension="public_signal", score=sentiment_score, status="assessed" if public_assessable else "not_assessable", confidence=round(confidence * 0.75, 3),
                       rationale="Qualitative public and independent evidence; it is weaker than primary reporting." if public_assessable else "No relevant independent or public signal was retrieved.", citations=sentiment_citations),
        DimensionScore(dimension="evidence_coverage", score=round(coverage * 100), confidence=dimension_confidence,
                       rationale="Diagnostic only: it measures source-plan coverage and is excluded from the overall score.",
                       citations=[item["evidence_id"] for item in evidence[:4]]),
    ]
    core_weights = {
        "material_sustainability": 0.35,
        "brand_commitments": 0.25,
        "certification_and_supply_chain": 0.25,
        "public_signal": 0.15,
    }
    assessed_weight = sum(core_weights[score.dimension] for score in scores if score.dimension in core_weights and score.status == "assessed")
    overall = round(
        sum(score.score * core_weights[score.dimension] for score in scores if score.dimension in core_weights and score.status == "assessed")
        / assessed_weight
    ) if assessed_weight else 0
    return scores, overall, round(confidence, 3)
