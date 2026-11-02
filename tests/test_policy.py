import pytest

from src.policy import detect_conflicts, score_assessment, sufficiency, validate_dimension_citations
from src.schemas import ProductSummary


def _item(evidence_id: str, category: str, text: str, **extra: object) -> dict:
    return {
        "evidence_id": evidence_id,
        "category": category,
        "text": text,
        "source_file": f"{evidence_id}.md",
        "credibility_tier": "primary" if category in {"brand", "materials"} else "secondary",
        "retrieval_score": 0.8,
        "recency_score": 0.8,
        "published_date": "2026-01-01",
        **extra,
    }


def test_sufficiency_keeps_original_required_sources_after_a_retry_plan() -> None:
    evidence = [_item("brand", "brand", "The brand reduced water use"), _item("material", "materials", "Cotton has impacts")]
    sufficient, missing, coverage = sufficiency(["brand_evidence", "material_evidence", "public_signal"], evidence)
    assert not sufficient
    assert missing == ["public_signal"]
    assert coverage == pytest.approx(2 / 3, abs=0.0001)


def test_conflicts_require_same_claim_dimension() -> None:
    product = ProductSummary(query="Levi's jeans", brand="Levi's")
    official = _item("official", "brand", "Levi's reduced water use in finishing.")
    labour_criticism = _item("labour", "independent", "Critics raised forced labor concerns in the supply chain.")
    water_criticism = _item("water", "independent", "Critics called the brand's water claims misleading and pollution remains a concern.")
    assert detect_conflicts(product, [official, labour_criticism]) == []
    conflicts = detect_conflicts(product, [official, water_criticism])
    assert len(conflicts) == 1
    assert conflicts[0].dimension == "water"


def test_coverage_does_not_change_overall_score() -> None:
    product = ProductSummary(query="Cotton tee", brand="Brand", materials=["cotton"])
    evidence = [
        _item("brand", "brand", "Brand made progress on climate targets."),
        _item("material", "materials", "Cotton can require water."),
        _item("public", "independent", "An independent source discusses the brand."),
    ]
    _, overall_low, _ = score_assessment(product, evidence, [], coverage=0.25)
    _, overall_high, _ = score_assessment(product, evidence, [], coverage=1.0)
    assert overall_low == overall_high


def test_unknown_material_is_not_assessable_and_cannot_use_generic_material_evidence() -> None:
    product = ProductSummary(query="Patagonia outerwear", brand="Patagonia", category="jacket", missing_attributes=["materials"])
    evidence = [_item("brand", "brand", "Patagonia reduced water use in its supply chain.")]
    scores, _, _ = score_assessment(product, evidence, [], coverage=1.0)
    material = next(score for score in scores if score.dimension == "material_sustainability")
    assert material.status == "not_assessable"
    assert material.score == 0
    assert material.citations == []


def test_dimension_citations_must_match_their_evidence_category() -> None:
    evidence = [_item("brand", "brand", "Brand reduced water use."), _item("material", "materials", "Cotton has water impacts.")]
    valid, invalid = validate_dimension_citations(
        [{"dimension": "material_sustainability", "status": "assessed", "citations": ["brand"]}],
        evidence,
    )
    assert not valid
    assert invalid == ["material_sustainability: brand"]
