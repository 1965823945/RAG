"""Generate sample PDFs for demo purposes."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_sample_pdfs(output_dir: str = "docs/pdfs", num_docs: int = 10):
    """Generate sample PDF documents for RAG demo.

    Args:
        output_dir: Directory to save generated PDFs
        num_docs: Number of sample documents to generate
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_topics = [
        (
            "Introduction to RAG",
            "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based models with generative models to improve the quality and relevance of generated text.",
        ),
        (
            "LangChain Basics",
            "LangChain is a framework for developing applications powered by large language models. It provides tools for chaining components together.",
        ),
        (
            "Vector Databases",
            "Vector databases are specialized databases that store and query high-dimensional vector embeddings, enabling efficient similarity search.",
        ),
        (
            "Text Embeddings",
            "Text embeddings are numerical representations of text that capture semantic meaning. They transform text into vectors of numbers.",
        ),
        (
            "Document Chunking",
            "Document chunking is the process of splitting large documents into smaller, manageable pieces for processing.",
        ),
        (
            "Semantic Search",
            "Semantic search understands the meaning behind queries, not just keyword matching, to return more relevant results.",
        ),
        (
            "Prompt Engineering",
            "Prompt engineering is the art of crafting effective prompts to get desired outputs from language models.",
        ),
        (
            "Fine-tuning LLMs",
            "Fine-tuning adapts pre-trained language models to specific tasks or domains for improved performance.",
        ),
        (
            "Retrieval Systems",
            "Retrieval systems find relevant documents from a large corpus based on user queries.",
        ),
        (
            "Generative AI",
            "Generative AI refers to AI systems that can create new content, including text, images, audio, and video.",
        ),
    ]

    for i in range(1, num_docs + 1):
        topic, content = sample_topics[i % len(sample_topics)]
        filename = f"{output_dir}/sample_{i}.pdf"

        c = canvas.Canvas(filename, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, f"Document {i}: {topic}")

        c.setFont("Helvetica", 12)
        text_lines = []
        words = content.split()
        line = ""
        for word in words:
            if len(line + word) < 80:
                line += word + " "
            else:
                text_lines.append(line)
                line = word + " "
        text_lines.append(line)

        y = 700
        for line in text_lines:
            c.drawString(50, y, line.strip())
            y -= 20

        c.save()
        print(f"Generated: {filename}")

    print(f"\nSuccessfully generated {num_docs} sample PDFs in {output_dir}")


if __name__ == "__main__":
    generate_sample_pdfs()
