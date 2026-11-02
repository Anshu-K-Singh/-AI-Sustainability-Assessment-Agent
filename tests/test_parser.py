from src.parser import parse_product
from src.schemas import AssessmentRequest


def test_parser_infers_brand_category_and_denim_material() -> None:
    product = parse_product(AssessmentRequest(query="Levi's denim jeans", depth="standard"))
    assert product.brand == "Levi's"
    assert product.category == "jeans"
    assert "denim" in product.materials
    assert "cotton" in product.materials
    assert not product.missing_attributes


def test_parser_reports_missing_brand() -> None:
    product = parse_product(AssessmentRequest(query="100% cotton t-shirt", depth="quick"))
    assert product.brand is None
    assert "brand" in product.missing_attributes
    assert product.materials == ["cotton"]


def test_parser_resolves_catalog_product_id_without_query() -> None:
    product = parse_product(AssessmentRequest(product_id="levis-501", depth="standard"))
    assert product.query == "Levi's 501 denim jeans"
    assert product.brand == "Levi's"
    assert product.category == "jeans"
    assert {"cotton", "denim"}.issubset(product.materials)
