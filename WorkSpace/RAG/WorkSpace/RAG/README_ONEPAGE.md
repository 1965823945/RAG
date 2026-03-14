Private Demo Domain RAG – Quick Start

Two launch options:
- Docker Compose (one-click): docker-compose up --build -d
- Local launcher: Run run_all.py (auto-dinds Docker availability)

Prereqs:
- Docker (optional, for one-click startup)
- Python 3.11+ (for local startup)

Endpoints:
- UI: http://localhost:8501
- API: http://localhost:8000 (POST /query)

How to test quickly:
- Ensure docs/domain_private_rag.pdf exists (or generate with domain_demo_pdf.py)
- Docker: docker-compose up --build -d, then visit UI/API
- Local: python run_all.py, then visit UI and API

Notes:
- This is a minimal, local-first prototype. You can swap in real embedding/LLMs later.
