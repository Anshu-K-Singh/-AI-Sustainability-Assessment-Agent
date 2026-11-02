from src.llm import LLMSettings, LocalLLM


def _llm_with_response(response: dict) -> LocalLLM:
    llm = LocalLLM(LLMSettings(True, "http://unused/v1", "test", "test", 1, 1))
    llm._json_completion = lambda *_args, **_kwargs: response  # type: ignore[method-assign]
    return llm


def test_llm_query_fallback_cannot_invent_scoring_materials() -> None:
    llm = _llm_with_response(
        {"brand": "Patagonia", "category": "jacket", "materials": ["recycled polyester"], "sustainability_keywords": ["recycled"]}
    )
    hints = llm.extract_product_hints(
        "Is this eco-conscious Patagonia outerwear worth buying?",
        allowed_brands=["Patagonia"],
        allowed_categories=["jacket"],
        allowed_materials=["recycled polyester"],
        allowed_keywords=["recycled"],
    )
    assert hints is not None
    assert hints.brand == "Patagonia"
    assert hints.category == "jacket"
    assert hints.materials == []
    assert hints.sustainability_keywords == []


def test_llm_synthesis_rejects_an_unknown_evidence_id() -> None:
    llm = _llm_with_response(
        {
            "overall_assessment": "Unsupported citation.",
            "main_findings": [{"finding": "Bad ID", "evidence_ids": ["NOT_RETRIEVED"], "category": "brand", "polarity": "mixed"}],
            "recommended_next_actions": [],
        }
    )
    result = llm.synthesize(
        product={}, evidence=[{"evidence_id": "E1", "category": "brand", "text": "Evidence."}], scores=[],
        overall_score=50, confidence=0.5, contradictions=[], uncertainties=[], allowed_categories=["brand"],
    )
    assert result is None


def test_llm_synthesis_accepts_a_grounded_citation() -> None:
    llm = _llm_with_response(
        {
            "overall_assessment": "Evidence is mixed.",
            "main_findings": [{"finding": "A grounded claim.", "evidence_ids": ["E1"], "category": "brand", "polarity": "mixed"}],
            "recommended_next_actions": ["Verify the product certificate."],
        }
    )
    result = llm.synthesize(
        product={}, evidence=[{"evidence_id": "E1", "category": "brand", "text": "Evidence."}], scores=[],
        overall_score=50, confidence=0.5, contradictions=[], uncertainties=[], allowed_categories=["brand"],
    )
    assert result is not None
    assert result.main_findings[0].evidence_ids == ["E1"]
