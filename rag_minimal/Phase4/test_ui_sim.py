"""UI Simulation Tests for Phase4: validate user input handling via backend.
This avoids launching a real browser while still exercising the UI interaction path.
"""
from rag_minimal.Phase4.streamlit_app import simulate_ui


def test_ui_sim_basic_input():
    q = "What is RAG?"
    resp = simulate_ui(q)
    assert isinstance(resp, str)
    assert len(resp) > 0
    assert "Prompt:" in resp or "prompt" in resp.lower()


def test_ui_sim_another_input():
    q = "Explain LangChain briefly"
    resp = simulate_ui(q)
    assert isinstance(resp, str)
    assert len(resp) > 0
