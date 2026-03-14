"""Helper to generate 10 sample PDFs for Day 2 demo."""
def main():
    try:
        from rag_minimal.day2 import generate_sample_pdfs
        generate_sample_pdfs(10)
        print("Generated 10 sample PDFs under docs/pdfs/")
    except Exception as e:
        print(f"Failed to generate samples: {e}")


if __name__ == "__main__":
    main()
