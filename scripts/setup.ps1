# AI Narrator - Environment setup script (PowerShell)
# Usage: .\scripts\setup.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "=== AI Narrator Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

# --- 1. Check Python 3.12 ---
$Python = $null
foreach ($candidate in @("python3.12", "python")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "3\.12\.\d+") {
            $Python = $candidate
            break
        }
    } catch { }
}

# Also check common Windows install paths
if (-not $Python) {
    $winPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Python312\python.exe"
    )
    foreach ($p in $winPaths) {
        if (Test-Path $p) {
            $ver = & $p --version 2>&1
            if ($ver -match "3\.12\.\d+") {
                $Python = $p
                break
            }
        }
    }
}

if (-not $Python) {
    Write-Host "ERROR: Python 3.12.x not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.12.9 first:"
    Write-Host "  https://www.python.org/downloads/release/python-3129/"
    exit 1
}

Write-Host "[1/6] Python found: $(& $Python --version)" -ForegroundColor Green

# --- 2. Create virtual environment ---
if (Test-Path ".venv") {
    Write-Host "[2/6] Virtual environment already exists, skipping." -ForegroundColor Green
} else {
    Write-Host "[2/6] Creating virtual environment..."
    & $Python -m venv .venv
}

# Activate venv
& .\.venv\Scripts\Activate.ps1

Write-Host "     Using: $(python --version) at $(Get-Command python | Select-Object -ExpandProperty Source)"

# --- 3. Upgrade pip ---
Write-Host "[3/6] Upgrading pip..."
python -m pip install --upgrade pip --quiet

# --- 4. Install VoxCPM ---
$voxcpmInstalled = python -c "import voxcpm" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[4/6] VoxCPM already installed, skipping." -ForegroundColor Green
} else {
    Write-Host "[4/6] Installing VoxCPM (editable)..."
    try {
        pip install -e ./VoxCPM --quiet
    } catch {
        Write-Host "     Editable install failed, trying regular install..."
        pip install ./VoxCPM --quiet
    }
}

# --- 5. Install dependencies ---
Write-Host "[5/6] Installing dependencies..."
pip install -r requirements.txt --quiet

# --- 6. Install CUDA PyTorch (if NVIDIA GPU detected) ---
try {
    $null = Get-Command nvidia-smi -ErrorAction Stop
    Write-Host "[6/6] NVIDIA GPU detected, installing CUDA PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --quiet
    pip install "fsspec<=2025.3.0" --quiet
} catch {
    Write-Host "[6/6] No NVIDIA GPU detected, using CPU PyTorch."
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next step: download model weights"
Write-Host "  python -c `"from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='./models/VoxCPM2')`""
