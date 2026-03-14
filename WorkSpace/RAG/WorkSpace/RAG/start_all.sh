#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[start_all] Root: $ROOT_DIR"
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
  echo "[start_all] Docker detected. Starting services via docker-compose..."
  docker-compose -f "$ROOT_DIR/docker-compose.yml" up --build -d
  echo "[start_all] Services started. Access UI: http://localhost:8501, API: http://localhost:8000"
else
  echo "[start_all] Docker not found. Starting locally via run_all.py..."
  python "$ROOT_DIR/run_all.py"
fi
