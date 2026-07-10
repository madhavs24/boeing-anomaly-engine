# One-time setup for the Boeing Anomaly Engine (Windows PowerShell).
# Run from inside the boeing-anomaly-engine folder:  .\setup.ps1

Write-Host "Setting up Boeing Anomaly Engine..." -ForegroundColor Cyan

# Create a local virtual environment (keeps deps isolated)
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Created .venv"
}

# Activate and install
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nDone. Next steps:" -ForegroundColor Green
Write-Host "  1) Fetch real Boeing data (one time, needs internet):"
Write-Host "     .\.venv\Scripts\python.exe -m src.data live"
Write-Host "  2) Run the engine:"
Write-Host "     .\.venv\Scripts\python.exe -m src.run"
Write-Host "  3) Run the auto-tuner:"
Write-Host "     .\.venv\Scripts\python.exe -m src.experiment"
