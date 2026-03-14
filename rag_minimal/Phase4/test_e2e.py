"""End-to-end test for Phase 4: verify integration via Phase3's answer_question."""
from rag_minimal.Phase3.day3 import answer_question


def main():
    q = "What is retrieval augmented generation?"
    ans = answer_question(q)
    assert isinstance(ans, str)
    assert len(ans) > 0
    print("Phase4 end-to-end test passed.")


if __name__ == "__main__":
    main()
