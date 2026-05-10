#!/usr/bin/env bash
# IRCS setup script – run once on a fresh Raspberry Pi OS installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== IRCS Setup ==="

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/5] Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    libgpiod2 \
    libopencv-dev \
    i2c-tools \
    git

# Enable I2C and SPI interfaces
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo "[2/5] Creating Python virtual environment …"
python3 -m venv "${SCRIPT_DIR}/.venv"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo "[3/5] Installing Python dependencies …"
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ── 4. Data directory and database bootstrap ───────────────────────────────────
echo "[4/5] Preparing data directory …"
mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${SCRIPT_DIR}/ml"

# Bootstrap the SQLite database by importing the logger
python3 - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath("$SCRIPT_DIR")))
from database.logger import DatabaseLogger
DatabaseLogger()          # creates tables if they don't exist
print("  Database initialised.")
EOF

# ── 5. Environment file ────────────────────────────────────────────────────────
echo "[5/5] Creating .env template (edit before running) …"
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    cat > "${ENV_FILE}" <<'ENVEOF'
# IRCS environment variables
# Copy this file to .env and fill in your values.

OPENAI_API_KEY=your-openai-api-key-here
ENVEOF
    echo "  .env template created at ${ENV_FILE}"
else
    echo "  .env already exists – skipping."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment with:"
echo "  source ${SCRIPT_DIR}/.venv/bin/activate"
echo "Then start the system with:"
echo "  python3 ${SCRIPT_DIR}/main.py"
