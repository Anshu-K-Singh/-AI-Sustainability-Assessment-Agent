import src.main as api


def test_openapi_exposes_health_assessment_and_workflow() -> None:
    schema = api.app.openapi()
    assert schema["info"]["title"] == "AI Sustainability Assessment Agent"
    assert "/health" in schema["paths"]
    assert "/assessments" in schema["paths"]
    assert "/workflow" in schema["paths"]
    post = schema["paths"]["/assessments"]["post"]
    assert post["requestBody"]["required"] is True
    assert "200" in post["responses"]
