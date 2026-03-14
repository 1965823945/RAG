"""Cross-platform launcher to start UI/API either via Docker Compose or locally."""
import subprocess
import shutil
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCKER_COMPOSE = os.path.join(ROOT, 'docker-compose.yml')

def docker_available():
    return shutil.which('docker') is not None and shutil.which('docker-compose') is not None

def start_docker():
    print('Starting docker-compose...')
    subprocess.Popen(['docker-compose', '-f', DOCKER_COMPOSE, 'up', '--build', '-d'], cwd=ROOT)
    print('Docker services started.')

def start_local():
    print('Starting local UI and API (no Docker)...')
    ui = subprocess.Popen(['streamlit', 'run', 'private_demo_domain/streamlit_app.py'], cwd=ROOT)
    api = subprocess.Popen(['uvicorn', 'private_demo_domain.api.main:app', '--host', '0.0.0.0', '--port', '8000'], cwd=ROOT)
    ui.wait()
    api.wait()

def main():
    if docker_available():
        start_docker()
    else:
        start_local()

if __name__ == '__main__':
    main()
