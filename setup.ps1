# =============================================================
# Evidence AI - one-time ONLINE PREPARATION
# =============================================================
# Creates the venv, installs dependencies, checks external tools,
# and downloads every local model. After this succeeds the app runs
# with no internet.
#
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# =============================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"

function Say($text)  { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Good($text) { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Bad($text)  { Write-Host "  [FAIL] $text" -ForegroundColor Red }

Say "Python"
$version = (python --version 2>&1)
if ($LASTEXITCODE -ne 0) {
    Bad "Python not found on PATH. Install Python 3.11+ from python.org."
    exit 1
}
Good $version

Say "Virtual environment"
if (-not (Test-Path $py)) {
    python -m venv (Join-Path $root ".venv")
    Good "Created .venv"
} else {
    Good ".venv already exists"
}
& $py -m pip install --upgrade pip setuptools wheel --quiet
Good "pip upgraded"

Say "PyTorch (CPU wheel)"
# The 4GB GPU is reserved for Ollama, so torch must be the CPU build.
& $py -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
if ($LASTEXITCODE -eq 0) { Good "torch (CPU) installed" } else { Bad "torch install failed"; exit 1 }

Say "Python dependencies"
& $py -m pip install --quiet -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -eq 0) { Good "requirements.txt installed" } else { Bad "dependency install failed"; exit 1 }

Say "Import verification"
& $py (Join-Path $root "scripts\verify_env.py")
if ($LASTEXITCODE -ne 0) { Bad "One or more imports failed - see above"; exit 1 }

Say "Tesseract OCR"
$tesseract = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseract) {
    Good ((& $tesseract --version 2>&1 | Select-Object -First 1))
} elseif (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Good "tesseract found on PATH"
} else {
    Warn "Tesseract missing - image OCR will be unavailable."
    Warn "Install with:  winget install UB-Mannheim.TesseractOCR"
}

Say "LibreOffice (legacy .doc support only)"
if (Test-Path "C:\Program Files\LibreOffice\program\soffice.exe") {
    Good "LibreOffice present - .doc conversion enabled"
} else {
    Warn "LibreOffice missing - .doc files cannot be ingested."
    Warn "All other formats work. Install with:"
    Warn "  winget install TheDocumentFoundation.LibreOffice"
}

Say "Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Bad "Ollama not installed. Get it from https://ollama.com/download"
    exit 1
}
Good (ollama --version)

$models = (ollama list | Out-String)
if ($models -match "qwen2.5vl:3b") {
    Good "qwen2.5vl:3b already pulled"
} else {
    Write-Host "  Pulling qwen2.5vl:3b (~3.2 GB, one time)..."
    ollama pull qwen2.5vl:3b
    if ($LASTEXITCODE -eq 0) { Good "qwen2.5vl:3b pulled" } else { Bad "pull failed"; exit 1 }
}

Say "Downloading local models (MiniLM, CrossEncoder, Whisper)"
& $py (Join-Path $root "scripts\prepare_models.py")
if ($LASTEXITCODE -ne 0) { Bad "Model preparation failed"; exit 1 }

Say "Demo data"
& $py (Join-Path $root "create_demo_data.py")

Write-Host "`n=============================================================" -ForegroundColor Green
Write-Host " SETUP COMPLETE" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " Next:"
Write-Host "   .\start.ps1                       launch API + UI"
Write-Host "   .venv\Scripts\python evaluation.py    run the evaluation"
Write-Host "   .venv\Scripts\python scripts\verify_offline.py   prove offline"
Write-Host ""
Write-Host " You can now disconnect from the internet." -ForegroundColor Cyan
