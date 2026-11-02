"""Source-specific retrieval adapters exposed to the LangGraph planner."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import is_claim_specific
from .retrieval import HybridRetriever, RetrievedEvidence
from .schemas import ProductSummary


@dataclass(frozen=True)
class RetrievalAction:
    tool: str
    query: str
    reason: str


class RetrievalTools:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def brand_evidence(self, product: ProductSummary, query: str, top_k: int) -> list[RetrievedEvidence]:
        if not product.brand:
            return []
        candidates = self.retriever.search(query, categories=["brand"], entities=[product.brand], top_k=top_k * 4)
        return [item for item in candidates if is_claim_specific(item.text, item.section)][:top_k]

    def material_evidence(self, product: ProductSummary, query: str, top_k: int) -> list[RetrievedEvidence]:
        material_text = " ".join(product.materials).lower()
        entities: list[str] = []
        if "cotton" in material_text or "denim" in material_text:
            entities.append("Cotton")
        if "polyester" in material_text:
            entities.append("Polyester")
        if "recycled" in material_text:
            entities.append("Recycled materials")
        # Never use generic cotton/polyester guidance for a product whose
        # composition is unknown. It would look grounded while answering a
        # question the evidence cannot support.
        if not entities:
            return []
        return self.retriever.search(
            query,
            categories=["materials"],
            entities=entities,
            top_k=top_k,
        )

    def certification_evidence(self, product: ProductSummary, query: str, top_k: int) -> list[RetrievedEvidence]:
        return self.retriever.search(query, categories=["certifications"], top_k=top_k)

    def public_signal(self, product: ProductSummary, query: str, top_k: int) -> list[RetrievedEvidence]:
        if not product.brand:
            return []
        candidates = self.retriever.search(
            query,
            categories=["sentiment", "independent"],
            entities=[product.brand],
            top_k=top_k * 3,
        )
        return [item for item in candidates if is_claim_specific(item.text, item.section)][:top_k]

    def execute(self, action: RetrievalAction, product: ProductSummary, top_k: int) -> list[RetrievedEvidence]:
        method = getattr(self, action.tool, None)
        if method is None:
            raise ValueError(f"Unknown retrieval tool: {action.tool}")
        return method(product, action.query, top_k)
