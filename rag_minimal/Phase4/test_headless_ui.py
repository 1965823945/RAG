"""Headless UI test for Phase4: validate UI startup in non-visual mode."""
import time
import subprocess
import requests


def test_headless_ui_starts_and_responds():
    port = 8502
    cmd = [
        "streamlit", "run", "rag_minimal/Phase4/streamlit_app.py",
        f"--server.port={port}", "--server.headless=true", "--server.enableCORS=false",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    started = False
    url = f"http://localhost:{port}/"
    start = time.time()
    try:
        while time.time() - start < 60:
            try:
                r = requests.get(url, timeout=2)
                if r.status_code < 500:
                    started = True
                    break
            except Exception:
                pass
            time.sleep(1)
        assert started, "Phase4 UI did not start in headless mode within timeout."
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
