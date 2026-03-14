import pytest


def test_all_phases_execution():
    # Phase 1: Run Phase1 implementation to ensure local behavior
    from rag_minimal.Phase1.day1 import day1_hello_world
    r1 = day1_hello_world()
    assert isinstance(r1, (str, type(None)))

    # Phase 2: Build vector store (loads PDFs from docs/pdfs or generates samples)
    from rag_minimal.Phase2.day2 import build_vector_store
    try:
        _ = build_vector_store()
    except Exception as e:
        pytest.fail(f"Phase2 failed: {e}")

    # Phase 3: Question answering using local LLM
    from rag_minimal.Phase3.day3 import answer_question
    r3 = answer_question("What is RAG?")
    assert isinstance(r3, str)

    # Phase 4: UI simulate path (no real browser required)
    from rag_minimal.Phase4.streamlit_app import simulate_ui
    ui_out = simulate_ui("Explain LangChain briefly.")
    assert isinstance(ui_out, str)
