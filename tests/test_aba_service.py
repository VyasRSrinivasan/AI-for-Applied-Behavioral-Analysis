from src.aba_service import retrieve_top_k, create_baseline_response, route_user_input


def test_retrieve_top_k_returns_results():
    results = retrieve_top_k("I feel anxious and overwhelmed", k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all("Antecedent" in item for item in results)


def test_baseline_response_includes_suggestions():
    retrieval = retrieve_top_k("I feel anxious and overwhelmed", k=2)
    response = create_baseline_response(retrieval)
    assert "suggestions" in response.lower() or "-" in response


def test_route_user_input_safe_returns_route():
    result = route_user_input("I am feeling a little stressed but okay.", k=1, mode="baseline", use_llm=False)
    assert result["route"] in {"baseline", "aba_rag"}
    assert "response" in result
