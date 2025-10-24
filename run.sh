#!/usr/bin/env bash
set -euo pipefail

# Resolve a working Python 3 (prefer Homebrew python3 if present)
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then
  PY_BIN="/opt/homebrew/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  echo "[ERROR] python3 not found. Please install Python 3 (e.g., 'brew install python')."
  exit 1
fi

echo "[INFO] Using Python: $($PY_BIN -V) at $PY_BIN"

# Create and activate virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment at $VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip & install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ensure data dir exists
mkdir -p data
mkdir -p figures

echo "[INFO] Environment ready. Running figure/table generator..."
python Mypaperfiguretable.py || {
  echo "[ERROR] Failed to run Mypaperfiguretable.py. Check that your CSVs exist under ./data/ and columns include '처방아이디' and '약재한글명'."
  exit 1
}

echo "[OK] Done. Figures saved under ./figures/, tables printed above."
