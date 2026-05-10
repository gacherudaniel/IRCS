#!/usr/bin/env bash
# IRCS setup script – run on a pre-configured Raspberry Pi.
# Assumes: OS packages, I2C, and camera are already enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== IRCS Setup ==="

# ── 1. Python virtual environment ─────────────────────────────────────────────
echo "[1/3] Creating Python virtual environment …"
python3 -m venv "${SCRIPT_DIR}/.venv"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo "[2/3] Installing Python dependencies …"
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ── 3. Data directories, database bootstrap, and .env ─────────────────────────
echo "[3/3] Preparing data directory and database …"
mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${SCRIPT_DIR}/ml"

# Bootstrap the SQLite database
python3 - <<EOF
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
from database.logger import DatabaseLogger
DatabaseLogger()
print("  Database initialised.")
EOF

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    cat > "${ENV_FILE}" <<'ENVEOF'
# IRCS environment variables
OPENAI_API_KEY=your-openai-api-key-here
ENVEOF
    echo "  .env template created – fill in your OpenAI key before running."
else
    echo "  .env already exists – skipping."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment with:"
echo "  source ${SCRIPT_DIR}/.venv/bin/activate"
echo "Then start the system with:"
echo "  python3 ${SCRIPT_DIR}/main.py"
