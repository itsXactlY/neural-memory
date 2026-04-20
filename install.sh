#!/bin/bash
#
# Neural Memory Adapter — Installer (2026-04-20)
#
# From-scratch lessons learned:
#   1. FastEmbed (ONNX) > sentence-transformers (no PyTorch conflict)
#   2. GPU recall via torch CUDA >> C++ bridge (Hopfield was biased)
#   3. SQLite = Source of Truth, MSSQL = optional mirror
#   4. pip install torch --index-url for CUDA support
#   5. numpy must be installed BEFORE FastEmbed
#   6. venv detection: check hermes-agent venv, then system
#
# Usage:
#   bash install.sh                         # auto-detect hermes-agent
#   bash install.sh /path/to/hermes-agent   # explicit path
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_ok()   { echo -e "${GREEN}✓${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
print_err()  { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${CYAN}→${NC} $1"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════╗"
echo "║   Neural Memory Adapter — Installer (2026-04)    ║"
echo "║   FastEmbed + GPU Recall + SQLite-First          ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# -------------------------------------------------------------------
# 1. Detect hermes-agent
# -------------------------------------------------------------------
HERMES_AGENT=""
if [ -n "$1" ]; then
    HERMES_AGENT="$1"
fi

if [ -z "$HERMES_AGENT" ]; then
    for candidate in \
        "$HOME/.hermes/hermes-agent" \
        "$HOME/hermes-agent" \
        "$HOME/.hermes/agent" \
        "/opt/hermes-agent" \
        "$HOME/projects/hermes-agent"; do
        if [ -d "$candidate" ] && [ -d "$candidate/plugins/memory" ]; then
            HERMES_AGENT="$candidate"
            break
        fi
    done
fi

if [ -z "$HERMES_AGENT" ] || [ ! -d "$HERMES_AGENT/plugins/memory" ]; then
    print_err "hermes-agent not found!"
    echo "  Checked: ~/.hermes/hermes-agent, ~/hermes-agent, ~/.hermes/agent"
    echo "  Install hermes-agent first, then run:"
    echo "    bash install.sh /path/to/hermes-agent"
    exit 1
fi

PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"
print_ok "hermes-agent: $HERMES_AGENT"
print_ok "Plugin target: $PLUGIN_DIR"

# -------------------------------------------------------------------
# 2. Python check
# -------------------------------------------------------------------
PYTHON=${PYTHON:-python3}
if ! $PYTHON --version &>/dev/null; then
    print_err "Python 3 not found"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1)
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
print_ok "Python: $PY_VER"

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    print_warn "Python 3.9+ recommended (you have $PY_VER)"
fi

# -------------------------------------------------------------------
# 3. Detect pip target
# -------------------------------------------------------------------
PIP=""
PIP_ARGS=""

# Priority: hermes-agent venv > active venv > system
if [ -f "$HERMES_AGENT/venv/bin/pip" ]; then
    PIP="$HERMES_AGENT/venv/bin/pip"
    PYTHON="$HERMES_AGENT/venv/bin/python3"
    print_info "Using hermes-agent venv: $HERMES_AGENT/venv"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/pip" ]; then
    PIP="$VIRTUAL_ENV/bin/pip"
    PYTHON="$VIRTUAL_ENV/bin/python3"
    print_info "Using active venv: $VIRTUAL_ENV"
else
    PIP="pip3"
    PIP_ARGS="--user"
    print_warn "No venv detected — using user install (--user)"
    print_warn "Consider creating a venv: python3 -m venv ~/.hermes/venv"
fi

# Upgrade pip (old pip causes many install failures)
print_info "Upgrading pip..."
$PIP install --quiet --upgrade pip 2>/dev/null || true

# -------------------------------------------------------------------
# 4. Core dependencies (ORDER MATTERS!)
# -------------------------------------------------------------------
print_info "Installing core dependencies..."

# 4a. numpy FIRST (FastEmbed needs it)
$PYTHON -c "import numpy" 2>/dev/null && print_ok "numpy" || {
    print_info "Installing numpy..."
    $PIP install $PIP_ARGS --quiet numpy
    print_ok "numpy installed"
}

# 4b. FastEmbed PRIMARY (ONNX, no PyTorch, ~50ms/emb)
#     Uses intfloat/multilingual-e5-large by default
$PYTHON -c "import fastembed" 2>/dev/null && print_ok "fastembed (ONNX backend)" || {
    print_info "Installing fastembed (primary embedding backend)..."
    print_info "  Model: intfloat/multilingual-e5-large (~500MB, auto-download)"
    $PIP install $PIP_ARGS --quiet fastembed
    print_ok "fastembed installed"
}

# 4c. Verify FastEmbed import works (model downloads on first use)
$PYTHON -c "from fastembed import TextEmbedding; print('  FastEmbed import OK')" 2>/dev/null && print_ok "FastEmbed ready (model downloads on first use)" || print_warn "FastEmbed import failed"

# -------------------------------------------------------------------
# 5. Optional: GPU recall engine (torch + CUDA)
# -------------------------------------------------------------------
echo ""
print_info "Checking GPU recall engine..."

HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    print_info "NVIDIA GPU detected (driver: $CUDA_VERSION)"

    # Check if torch with CUDA is already installed
    $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && {
        GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_ok "torch + CUDA: $GPU_NAME"
        HAS_CUDA=true
    } || {
        print_info "Installing torch with CUDA support..."
        print_info "  This may take a while (~2GB download)"

        # Detect CUDA version for correct index URL
        CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
        if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 12 ]; then
            # CUDA 12.x
            $PIP install $PIP_ARGS --quiet torch --index-url https://download.pytorch.org/whl/cu121
        elif [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 11 ]; then
            # CUDA 11.x
            $PIP install $PIP_ARGS --quiet torch --index-url https://download.pytorch.org/whl/cu118
        else
            # Fallback: CPU-only torch (smaller download)
            print_warn "Could not detect CUDA version — installing CPU-only torch"
            $PIP install $PIP_ARGS --quiet torch --index-url https://download.pytorch.org/whl/cpu
        fi

        # Verify
        $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && {
            GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            print_ok "torch + CUDA installed: $GPU_NAME"
            HAS_CUDA=true
        } || {
            print_warn "torch installed but CUDA not available — GPU recall disabled"
            print_warn "  Check: nvidia-smi works? CUDA toolkit installed?"
        }
    }
else
    print_warn "No NVIDIA GPU detected — using CPU recall (~500ms)"
    print_info "GPU recall optional — FastEmbed + numpy recall works fine for <50K memories"
fi

# -------------------------------------------------------------------
# 6. Optional: sentence-transformers (GPU batch embedding)
# -------------------------------------------------------------------
echo ""
$PYTHON -c "import sentence_transformers" 2>/dev/null && print_ok "sentence-transformers (optional, batch GPU embedding)" || {
    print_info "sentence-transformers not installed (optional)"
    print_info "  Only needed for GPU batch embedding or BAAI/bge-m3 model"
    print_info "  FastEmbed covers most use cases without PyTorch"
    print_info "  Install later: pip install sentence-transformers"
}

# -------------------------------------------------------------------
# 7. Optional: pyodbc (MSSQL mirror)
# -------------------------------------------------------------------
$PYTHON -c "import pyodbc" 2>/dev/null && print_ok "pyodbc (MSSQL available)" || {
    print_info "MSSQL not configured (optional — SQLite is primary)"
    print_info "  Install later: pip install pyodbc"
    print_info "  Also needs: ODBC Driver 18 + MSSQL Server"
}

# -------------------------------------------------------------------
# 8. Install plugin files
# -------------------------------------------------------------------
echo ""
print_info "Installing plugin files..."
mkdir -p "$PLUGIN_DIR"

# Core files
CORE_FILES=(
    __init__.py plugin.yaml config.py
    memory_client.py neural_memory.py
    embed_provider.py gpu_recall.py
    dream_engine.py dream_worker.py
    access_logger.py
    cpp_bridge.py cpp_dream_backend.py
    mssql_store.py dream_mssql_store.py
    lstm_knn_bridge.py test_suite.py
)

INSTALLED=0
for f in "${CORE_FILES[@]}"; do
    if [ -f "$PYTHON_DIR/$f" ]; then
        cp "$PYTHON_DIR/$f" "$PLUGIN_DIR/"
        ((INSTALLED++))
    elif [ -f "$SCRIPT_DIR/hermes-plugin/$f" ]; then
        cp "$SCRIPT_DIR/hermes-plugin/$f" "$PLUGIN_DIR/"
        ((INSTALLED++))
    fi
done

# Optional files
for f in fast_ops.pyx fast_ops.cpython*.so; do
    SRC=$(ls "$PYTHON_DIR/$f" 2>/dev/null | head -1)
    [ -n "$SRC" ] && cp "$SRC" "$PLUGIN_DIR/" 2>/dev/null || true
done

print_ok "Plugin files installed ($INSTALLED files)"

# -------------------------------------------------------------------
# 9. Initialize database
# -------------------------------------------------------------------
DB_PATH="${NEURAL_MEMORY_DB_PATH:-$HOME/.neural_memory/memory.db}"
mkdir -p "$(dirname "$DB_PATH")"

if [ ! -f "$DB_PATH" ]; then
    print_info "Creating database at $DB_PATH..."
    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
stats = s.stats()
s.close()
print(f'  Created: {stats}')
" 2>/dev/null && print_ok "Database initialized" || print_warn "Database will auto-create on first use"
else
    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
stats = s.stats()
s.close()
print(f'  {stats[\"memories\"]} memories, {stats[\"connections\"]} connections')
" 2>/dev/null && print_ok "Existing database found" || print_warn "Database may need repair"
fi

# -------------------------------------------------------------------
# 10. Configure Hermes (config.yaml)
# -------------------------------------------------------------------
CONFIG_FILE="$HOME/.hermes/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    if grep -q "provider: neural" "$CONFIG_FILE" 2>/dev/null; then
        print_ok "config.yaml: neural provider configured"
    else
        print_info "Updating config.yaml..."
        $PYTHON -c "
import yaml, os
config_file = '$CONFIG_FILE'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f) or {}

config.setdefault('memory', {})
config['memory']['provider'] = 'neural'
config['memory'].setdefault('neural', {})
config['memory']['neural']['db_path'] = '$DB_PATH'
config['memory']['neural']['embedding_backend'] = 'fastembed'

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('  config.yaml updated')
" 2>/dev/null && print_ok "config.yaml updated" || {
            print_warn "Could not auto-update config.yaml"
            echo "  Add this to $CONFIG_FILE:"
            echo ""
            echo "  memory:"
            echo "    provider: neural"
            echo "    neural:"
            echo "      db_path: $DB_PATH"
            echo "      embedding_backend: fastembed"
        }
    fi
else
    print_warn "config.yaml not found at $CONFIG_FILE"
    echo "  Create with:"
    echo ""
    echo "  memory:"
    echo "    provider: neural"
    echo "    neural:"
    echo "      db_path: $DB_PATH"
    echo "      embedding_backend: fastembed"
    echo "      dream:"
    echo "        enabled: true"
    echo "        idle_threshold: 600"
    echo "        memory_threshold: 50"
fi

# -------------------------------------------------------------------
# 11. Verify installation
# -------------------------------------------------------------------
echo ""
print_info "Verifying installation..."

$PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')

# Test imports
from memory_client import NeuralMemory
from embed_provider import get_embedding
print('  Imports: OK')

# Test embedding
emb = get_embedding('test memory content', backend='hash')
print(f'  Embedding: OK ({len(emb)}d)')

# Test memory system
m = NeuralMemory(embedding_backend='hash', use_cpp=False)
m.remember('installation test', label='test')
results = m.recall('installation test')
print(f'  Store+Recall: OK ({len(results)} results)')
m.close()
print('  All checks passed')
" 2>/dev/null && print_ok "Verification passed" || print_warn "Verification had warnings (may work at runtime)"

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Neural Memory Adapter installed!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo ""
echo "  Provider:   neural"
echo "  Backend:    FastEmbed (intfloat/multilingual-e5-large)"
echo "  Database:   $DB_PATH (SQLite)"
echo "  GPU Recall: $([ "$HAS_CUDA" = true ] && echo 'enabled (CUDA)' || echo 'disabled (CPU fallback)')"
echo "  Plugin:     $PLUGIN_DIR"
echo ""
echo "  Next steps:"
echo "    1. Restart hermes: hermes gateway restart"
echo "    2. Test: neural_remember / neural_recall"
echo ""
