Param()
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
  Write-Host "Starting Docker Compose..."
  docker-compose -f "$root\docker-compose.yml" up --build -d
  Write-Host "UI and API are starting. Access: UI http://localhost:8501, API http://localhost:8000"
}
else {
  Write-Host "Docker not found. Running local launcher..."
  python "$root\run_all.py"
}
