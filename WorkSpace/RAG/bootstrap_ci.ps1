Param()
    [string]$Action = "Up"
Write-Host "BOOTSTRAP: Root=$(Get-Location)"
$root = (Get-Location).Path
if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
  Write-Host "Starting docker-compose..."
  & docker-compose -f "$root\docker-compose.yml" up --build -d
  Write-Host "UI and API started via Docker Compose."
} else {
  Write-Host "docker-compose not found. Run UI and API locally:"
  Write-Host "UI: streamlit run .\WorkSpace\RAG\private_demo_domain\streamlit_app.py"
  Write-Host "API: uvicorn .\WorkSpace\RAG\private_demo_domain\api\main.py:app --host 0.0.0.0 --port 8000"
}
