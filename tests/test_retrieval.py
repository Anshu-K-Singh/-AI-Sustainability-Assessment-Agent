from src.retrieval import HybridRetriever


def test_brand_retrieval_preserves_citation(engine) -> None:
    results = engine.retriever.search(
        "Levi Strauss cotton water climate sustainability targets",
        categories=["brand"],
        entities=["Levi's"],
        top_k=3,
    )
    assert len(results) == 3
    assert all(item.category == "brand" for item in results)
    assert all(item.entity == "Levi's" for item in results)
    assert all(item.source_url.startswith("https://") for item in results)
    assert all(item.source_title and item.publisher and item.captured_at for item in results)
    assert all(item.retrieval_methods for item in results)


def test_cotton_retrieval_does_not_return_polyester(engine) -> None:
    results = engine.retriever.search(
        "cotton denim environmental impact water",
        categories=["materials"],
        entities=["Cotton"],
        top_k=5,
    )
    assert results
    assert {item.entity for item in results} == {"Cotton"}
