# Minimal RAG Demo

A concise, end-to-end Retrieval-Augmented Generation (RAG) workflow in Python using LangChain and ChromaDB.

## Features

- PDF document loading and processing
- Text chunking and embedding generation
- ChromaDB vector storage
- Simple LLM for demonstration (no API keys required)
- Streamlit web UI for interactive Q&A

## Project Structure

```
rag_minimal/
├── __init__.py         # Package initialization
├── main.py             # Main entry point - runs full demo
├── llm.py              # Simple LLM wrapper
├── embeddings.py       # Embeddings module
├── loader.py           # PDF document loader
├── chunker.py          # Text chunking
├── vectorstore.py      # ChromaDB vector store
├── retriever.py        # Retrieval module
├── chain.py            # RAG chain
├── app.py              # Streamlit web UI
└── generate_samples.py # Generate sample PDFs
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Generate sample PDFs:
```bash
python -m rag_minimal.generate_samples
```

2. Run the demo:
```bash
python -m rag_minimal.main
```

3. Launch the Streamlit UI:
```bash
streamlit run rag_minimal/app.py
```

## Running Tests

```bash
pytest --cov=rag_minimal --cov-report=xml
```

## Running CI

```bash
ruff check rag_minimal/
```

## Notes

- This is a minimal, self-contained example using local, offline components to avoid API keys.
- For production use, replace `SimpleLLM` and `FakeEmbeddings` with actual LLM providers and embedding models.
- The sample PDFs are automatically generated if none exist in `docs/pdfs/`.
