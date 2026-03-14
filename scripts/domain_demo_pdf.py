"""Utility to generate a tiny sample domain PDF for quick testing."""
def main():
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas("docs/domain_private_rag.pdf")
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "Domain: Python ▶ PyTorch ▶ HuggingFace")
        c.drawString(100, 730, "This is a minimal sample PDF for the private RAG demo.")
        c.save()
        print("Generated docs/domain_private_rag.pdf")
    except Exception as e:
        print(f"Could not generate sample PDF: {e}")

if __name__ == "__main__":
    main()
