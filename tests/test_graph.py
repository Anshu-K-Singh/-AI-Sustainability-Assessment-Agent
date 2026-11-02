from types import SimpleNamespace

from src.llm import ConflictReview
from src.schemas import AssessmentRequest, Contradiction


def test_standard_assessment_is_grounded_and_traceable(engine) -> None:
    response = engine.assess(
        AssessmentRequest(
            query="Levi's denim jeans",
            brand="Levi's",
            category="jeans",
            materials=["cotton", "denim"],
            depth="standard",
        )
    )
    assert response.product_summary.brand == "Levi's"
    assert response.evidence
    assert all(item.source_url for item in response.evidence)
    assert any(score.citations for score in response.scores)
    assert response.metadata.tool_calls == 3
    steps = [event.step for event in response.trace]
    assert steps == [
        "parse_input",
        "plan_sources",
        "retrieve_evidence",
        "check_sufficiency",
        "detect_conflicts",
        "llm_conflict_analysis",
        "score_assessment",
        "synthesize",
    ]


def test_unknown_brand_degrades_gracefully(engine) -> None:
    response = engine.assess(
        AssessmentRequest(query="100% cotton t-shirt from Brand X", materials=["cotton"], depth="standard")
    )
    assert response.product_summary.brand is None
    assert any("brand" in uncertainty.lower() for uncertainty in response.uncertainties)
    assert response.evidence


def test_unknown_material_does_not_retrieve_or_score_generic_materials(engine) -> None:
    response = engine.assess(AssessmentRequest(query="Patagonia everyday outerwear", depth="standard"))
    material = next(score for score in response.scores if score.dimension == "material_sustainability")
    assert "materials" in response.product_summary.missing_attributes
    assert material.status == "not_assessable"
    assert material.citations == []
    assert not any(item.category == "materials" for item in response.evidence)


def test_unconfirmed_llm_conflict_is_an_uncertainty_not_a_confirmed_contradiction(engine) -> None:
    class InsufficientConflictLLM:
        enabled = True
        settings = SimpleNamespace(max_conflict_pairs=1, model="test")

        @staticmethod
        def review_conflict(*_args, **_kwargs):
            return ConflictReview(decision="insufficient_evidence", topic="water", explanation="Evidence does not establish a direct conflict.")

    engine.llm = InsufficientConflictLLM()
    state = {
        "contradictions": [
            Contradiction(dimension="water", description="candidate", supporting_evidence=["official"], opposing_evidence=["external"]).model_dump()
        ],
        "evidence": [
            {"evidence_id": "official", "category": "brand", "text": "Brand reduced water use."},
            {"evidence_id": "external", "category": "independent", "text": "Critics call water claims misleading."},
        ],
        "uncertainties": [],
        "llm_roles_used": [],
        "trace": [],
    }
    result = engine._llm_conflict_analysis(state)
    assert result["contradictions"] == []
    assert "could not be semantically confirmed" in result["uncertainties"][0]
