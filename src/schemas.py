"""Public request/response contracts for the sustainability assessment API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AssessmentDepth(str, Enum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class AssessmentRequest(BaseModel):
    """A product query plus optional structured hints."""

    query: str = Field(default="", examples=["Levi's denim jeans"])
    product_id: str | None = Field(default=None, examples=["levis-501"])
    category: str | None = Field(default=None, examples=["jeans"])
    brand: str | None = Field(default=None, examples=["Levi's"])
    materials: list[str] = Field(default_factory=list, examples=[["cotton", "denim"]])
    depth: AssessmentDepth = AssessmentDepth.standard

    @field_validator("materials", mode="before")
    @classmethod
    def normalize_materials(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def require_query_or_product_id(self) -> "AssessmentRequest":
        if not self.query.strip() and not self.product_id:
            raise ValueError("Provide either a non-empty query or a product_id")
        return self


class ProductSummary(BaseModel):
    product_id: str | None = None
    query: str
    brand: str | None = None
    category: str | None = None
    materials: list[str] = Field(default_factory=list)
    sustainability_keywords: list[str] = Field(default_factory=list)
    missing_attributes: list[str] = Field(default_factory=list)
    extraction_method: str = "rules"


class EvidenceItem(BaseModel):
    evidence_id: str
    snippet: str
    source_file: str
    source_url: str
    source_title: str | None = None
    publisher: str | None = None
    published_date: str | None = None
    captured_at: str | None = None
    source_type: str
    category: str
    entity: str
    section: str = ""
    page: int | None = None
    credibility_tier: str
    retrieval_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    retrieval_methods: list[str] = Field(default_factory=list)


class DimensionScore(BaseModel):
    dimension: str
    score: int = Field(ge=0, le=100)
    status: Literal["assessed", "not_assessable"] = "assessed"
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    citations: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    dimension: str
    description: str
    supporting_evidence: list[str] = Field(default_factory=list)
    opposing_evidence: list[str] = Field(default_factory=list)
    severity: str = "medium"


class TraceEvent(BaseModel):
    step: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)


class AssessmentMetadata(BaseModel):
    depth: AssessmentDepth
    retrieval_rounds: int
    tool_calls: int
    evidence_count: int
    corpus_mode: str = "local"
    planner_mode: str = "simulated"
    llm_model: str | None = None
    llm_attempted: bool = False
    llm_roles_used: list[str] = Field(default_factory=list)


class GroundedFinding(BaseModel):
    """A short, evidence-linked statement shown in the assessment summary."""

    finding: str
    evidence_ids: list[str] = Field(min_length=1)
    category: str
    polarity: str = "mixed"


class AssessmentResponse(BaseModel):
    product_summary: ProductSummary
    scores: list[DimensionScore]
    overall_score: int = Field(ge=0, le=100)
    overall_assessment: str
    main_findings: list[GroundedFinding] = Field(default_factory=list)
    evidence: list[EvidenceItem]
    confidence: float = Field(ge=0.0, le=1.0)
    contradictions: list[Contradiction] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    trace: list[TraceEvent]
    metadata: AssessmentMetadata


class HealthResponse(BaseModel):
    status: str
    chunks: int
    indexes: dict[str, bool]
    llm: dict[str, str | bool]
