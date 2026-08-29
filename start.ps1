# =============================================================
# Evidence AI - start the application (OFFLINE RUNTIME)
# =============================================================
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
#   .\start.ps1 -ApiOnly     FastAPI only
#   .\start.ps1 -UiOnly      Streamlit only
# =============================================================

param(
    [switch]$ApiOnly,
    [switch]$UiOnly,
    [int]$ApiPort = 8000,
    [int]$UiPort  = 8501
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"

function Say($text)  { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Good($text) { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Bad($text)  { Write-Host "  [FAIL] $text" -ForegroundColor Red }

if (-not (Test-Path $py)) {
    Bad "No virtual environment found. Run .\setup.ps1 first."
    exit 1
}

Say "Preflight"

# Offline AI (Ollama). This is checked but NEVER fatal: Online mode does not
# need Ollama, and the app reports offline readiness in its own UI with a
# Check Again / Install / Switch to Online flow.
$ollamaReady = $false
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $ollamaReady = $true
    Good "Ollama is running"
} catch {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Warn "Ollama is not responding - starting it..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
            $ollamaReady = $true
            Good "Ollama started"
        } catch {
            Warn "Could not start Ollama. Offline mode will be unavailable."
        }
    } else {
        Warn "Ollama is not installed. Offline mode will be unavailable."
    }
}

if ($ollamaReady) {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    if ($tags.models.name -contains "qwen2.5vl:3b") {
        Good "Offline AI ready - qwen2.5vl:3b"
    } else {
        Warn "qwen2.5vl:3b not installed. Run: ollama pull qwen2.5vl:3b"
        $ollamaReady = $false
    }
}

# Online AI credentials. Presence only - the key itself is never printed.
if ($env:OPENAI_API_KEY -or (Test-Path (Join-Path $root ".env"))) {
    Good "Online AI credentials source present (.env or environment)"
} else {
    Warn "No OPENAI_API_KEY found. Online mode will need configuring in .env."
}

if (-not $ollamaReady) {
    Warn "Offline AI is not ready. The app will still start - select Online"
    Warn "in the sidebar, or use the setup buttons shown in the UI."
}

# Tesseract
if (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe") {
    Good "Tesseract OCR available"
} else {
    Warn "Tesseract missing - images will be indexed without OCR text"
}

# Index
$dbPath = Join-Path $root "index\store.db"
if (Test-Path $dbPath) {
    Good "Persistent index found"
} else {
    Warn "No index yet - ingest files from the UI, or run:"
    Warn "  .venv\Scripts\python scripts\ingest_demo.py"
}

Say "Launching"

if (-not $UiOnly) {
    Start-Process -FilePath $py `
        -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
        -WorkingDirectory $root
    Good "FastAPI   http://127.0.0.1:$ApiPort/docs"
}

if (-not $ApiOnly) {
    Good "Streamlit http://localhost:$UiPort"
    Write-Host "`n  Press Ctrl+C to stop.`n" -ForegroundColor Yellow
    & $py -m streamlit run (Join-Path $root "app.py") --server.port $UiPort
} else {
    Write-Host "`n  API running. Press Ctrl+C to stop.`n" -ForegroundColor Yellow
    while ($true) { Start-Sleep -Seconds 3600 }
}
