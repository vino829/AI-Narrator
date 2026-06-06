#!/usr/bin/env bash
# AI Narrator - Environment setup script
# Usage: bash scripts/setup.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== AI Narrator Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# --- 1. Check Python 3.12 ---
PYTHON=""
for candidate in python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oP '3\.12\.\d+')
        if [ -n "$ver" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12.x not found."
    echo ""
    echo "Please install Python 3.12.9 first:"
    echo "  Windows: https://www.python.org/downloads/release/python-3129/"
    echo "  Linux:   sudo apt install python3.12 python3.12-venv"
    echo "  macOS:   brew install python@3.12"
    exit 1
fi

echo "[1/6] Python found: $($PYTHON --version)"

# --- 2. Create virtual environment ---
if [ -d ".venv" ]; then
    echo "[2/6] Virtual environment already exists, skipping."
else
    echo "[2/6] Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# Activate venv
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate   # Windows (Git Bash / MSYS2)
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate       # Linux / macOS
fi

echo "     Using: $(python --version) at $(which python)"

# --- 3. Upgrade pip ---
echo "[3/6] Upgrading pip..."
python -m pip install --upgrade pip --quiet

# --- 4. Install VoxCPM ---
if python -c "import voxcpm" &>/dev/null; then
    echo "[4/6] VoxCPM already installed, skipping."
else
    echo "[4/6] Installing VoxCPM (editable)..."
    if ! pip install -e ./VoxCPM --quiet 2>/dev/null; then
        echo "     Editable install failed, trying regular install..."
        pip install ./VoxCPM --quiet
    fi
fi

# --- 5. Install dependencies ---
echo "[5/6] Installing dependencies..."
pip install -r requirements.txt --quiet

# --- 6. Install CUDA PyTorch (if NVIDIA GPU detected) ---
if command -v nvidia-smi &>/dev/null; then
    echo "[6/6] NVIDIA GPU detected, installing CUDA PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --quiet 2>&1 | grep -v "dependency conflicts"
    pip install "fsspec<=2025.3.0" --quiet
else
    echo "[6/6] No NVIDIA GPU detected, using CPU PyTorch."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment:"
if [ -f ".venv/Scripts/activate" ]; then
    echo "  source .venv/Scripts/activate   # Git Bash"
    echo "  .venv\\Scripts\\activate          # CMD / PowerShell"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "Next step: download model weights"
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='./models/VoxCPM2')\""
