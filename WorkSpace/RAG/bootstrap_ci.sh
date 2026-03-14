#!/usr/bin/env bash
set -e
ROOT_DIR=$(cd "$(dirname "$0")"/..; pwd)
echo "[BOOTSTRAP] Root: $ROOT_DIR"
if command -v docker-compose >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Starting docker-compose (UI + API)"
  docker-compose -f "$ROOT_DIR/docker-compose.yml" up --build -d
  echo "[BOOTSTRAP] Services started. Access: UI http://localhost:8501, API http://localhost:8000"
else
  echo "[BOOTSTRAP] docker-compose not found. Running local UI and API."
  echo "To run UI locally: streamlit run $ROOT_DIR/WorkSpace/RAG/private_demo_domain/streamlit_app.py"
  echo "To run API locally: uvicorn $ROOT_DIR/WorkSpace/RAG/private_demo_domain/api/main.py:app --host 0.0.0.0 --port 8000"
fi
