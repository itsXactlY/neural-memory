#!/bin/bash
#
# Neural Memory Adapter — Installer (2026-04)
#
# Installs the neural memory plugin into hermes-agent.
# Learned from production: FastEmbed > sentence-transformers for CPU,
# GPU recall via torch CUDA, SQLite as source of truth.
#
# Usage:
#   bash install.sh                         # auto-detect hermes-agent
#   bash install.sh /path/to/hermes-agent   # explicit path
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"

# Colors
GREEN='[0;32m'
YELLOW='[1;33m'
RED='[0;31m'
CYAN='[0;36m'
NC='[0m'
BOLD='[1m'

print_ok()   { echo -e "${GREEN}✓${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
print_err()  { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${CYAN}→${NC} $1"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   Neural Memory Adapter — Installer          ║"
echo "║   FastEmbed + GPU Recall + SQLite            ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# -------------------------------------------------------------------
# 1. Detect hermes-agent
# -------------------------------------------------------------------
HERMES_AGENT=""
if [ -n "$1" ]; then
    HERMES_AGENT="$1"
fi

if [ -z "$HERMES_AGENT" ]; then
    for candidate in         "$HOME/.hermes/hermes-agent"         "$HOME/hermes-agent"         "/opt/hermes-agent"; do
        if [ -n "$candidate" ] && [ -d "$candidate/plugins/memory" ]; then
            HERMES_AGENT="$candidate"
            break
        fi
    done
fi

if [ -z "$HERMES_AGENT" ] || [ ! -d "$HERMES_AGENT/plugins/memory" ]; then
    print_err "hermes-agent not found!"
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
print_ok "Python: $PY_VER"

# -------------------------------------------------------------------
# 3. Dependencies (learned from production)
# -------------------------------------------------------------------
print_info "Installing dependencies..."

# Determine pip target (venv or system)
PIP="pip"
if [ -f "$HERMES_AGENT/venv/bin/pip" ]; then
    PIP="$HERMES_AGENT/venv/bin/pip"
    print_info "Using hermes-agent venv pip"
fi

# FastEmbed (PRIMARY — ONNX, no PyTorch conflict, ~50ms/emb)
$PYTHON -c "import fastembed" 2>/dev/null && print_ok "fastembed" || {
    print_info "Installing fastembed (ONNX embedding backend)..."
    $PIP install --quiet fastembed 2>/dev/null || $PIP install --user --quiet fastembed
    print_ok "fastembed installed"
}

# sentence-transformers (for GPU batch embedding)
$PYTHON -c "import sentence_transformers" 2>/dev/null && print_ok "sentence-transformers" || {
    print_info "Installing sentence-transformers (GPU batch embedding)..."
    $PIP install --quiet sentence-transformers 2>/dev/null || $PIP install --user --quiet sentence-transformers
    print_ok "sentence-transformers installed"
}

# torch (for GPU recall engine)
$PYTHON -c "import torch" 2>/dev/null && print_ok "torch" || {
    print_info "Installing torch (GPU recall engine)..."
    $PIP install --quiet torch 2>/dev/null || $PIP install --user --quiet torch
    print_ok "torch installed"
}

# numpy
$PYTHON -c "import numpy" 2>/dev/null && print_ok "numpy" || {
    print_info "Installing numpy..."
    $PIP install --quiet numpy 2>/dev/null || $PIP install --user --quiet numpy
    print_ok "numpy installed"
}

# pyodbc (optional, MSSQL)
$PYTHON -c "import pyodbc" 2>/dev/null && print_ok "pyodbc (MSSQL)" || print_warn "pyodbc not found — MSSQL unavailable (optional)"

# Cython (optional, fast_ops)
$PYTHON -c "import Cython" 2>/dev/null && print_ok "Cython" || print_warn "Cython not found — fast_ops build skipped (optional)"

# Check CUDA
$PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null &&     print_ok "CUDA available (GPU recall enabled)" ||     print_warn "No CUDA — GPU recall disabled (Python fallback, ~500ms)"

# -------------------------------------------------------------------
# 4. Install plugin files
# -------------------------------------------------------------------
print_info "Installing plugin files..."
mkdir -p "$PLUGIN_DIR"

# Core files (synced from python/ or hermes-plugin/)
for f in __init__.py plugin.yaml config.py memory_client.py embed_provider.py          neural_memory.py gpu_recall.py cpp_bridge.py cpp_dream_backend.py          mssql_store.py dream_mssql_store.py dream_engine.py dream_worker.py          access_logger.py lstm_knn_bridge.py test_suite.py README.md; do
    if [ -f "$PYTHON_DIR/$f" ]; then
        cp "$PYTHON_DIR/$f" "$PLUGIN_DIR/"
    elif [ -f "$SCRIPT_DIR/hermes-plugin/$f" ]; then
        cp "$SCRIPT_DIR/hermes-plugin/$f" "$PLUGIN_DIR/"
    fi
done

# fast_ops source
cp "$PYTHON_DIR/fast_ops.pyx" "$PLUGIN_DIR/" 2>/dev/null || true
# Copy pre-built .so if available
SO_FILE=$(ls "$PYTHON_DIR"/fast_ops.cpython*.so 2>/dev/null | head -1)
[ -n "$SO_FILE" ] && cp "$SO_FILE" "$PLUGIN_DIR/" 2>/dev/null || true

print_ok "Plugin files installed"

# -------------------------------------------------------------------
# 5. Build Cython fast_ops (optional)
# -------------------------------------------------------------------
if [ -f "$PYTHON_DIR/setup_fast.py" ] && $PYTHON -c "import Cython" 2>/dev/null; then
    print_info "Building Cython fast_ops..."
    cd "$PYTHON_DIR"
    if $PYTHON setup_fast.py build_ext --inplace 2>/dev/null; then
        SO_FILE=$(ls "$PYTHON_DIR"/fast_ops.cpython*.so 2>/dev/null | head -1)
        if [ -n "$SO_FILE" ]; then
            cp "$SO_FILE" "$PLUGIN_DIR/"
            print_ok "fast_ops compiled"
        fi
    else
        print_warn "fast_ops build failed — Python fallback"
    fi
fi

# -------------------------------------------------------------------
# 6. Build C++ library (optional, has Hopfield bias)
# -------------------------------------------------------------------
if [ -d "$SCRIPT_DIR/build" ] && [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    CPP_LIB="$SCRIPT_DIR/build/libneural_memory.so"
    if [ ! -f "$CPP_LIB" ]; then
        print_info "Building C++ library (optional, has Hopfield bias)..."
        cd "$SCRIPT_DIR/build"
        cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MSSQL=OFF 2>/dev/null &&         make neural_memory -j$(nproc) 2>/dev/null &&         print_ok "C++ bridge built" ||         print_warn "C++ build failed — use GPU recall instead"
    else
        print_ok "C++ bridge: $CPP_LIB"
    fi
fi

# -------------------------------------------------------------------
# 7. Initialize database
# -------------------------------------------------------------------
DB_PATH="${NEURAL_MEMORY_DB_PATH:-$HOME/.neural_memory/memory.db}"
mkdir -p "$(dirname "$DB_PATH")"

if [ ! -f "$DB_PATH" ]; then
    print_info "Initializing database at $DB_PATH..."
    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
print(f'  {s.stats()["memories"]} memories')
s.close()
" 2>/dev/null && print_ok "Database initialized" || print_warn "Database will auto-create on first use"
else
    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
stats = s.stats()
s.close()
print(f'  {stats["memories"]} memories, {stats["connections"]} connections')
" 2>/dev/null && print_ok "Existing database found" || print_warn "Database may need repair"
fi

# -------------------------------------------------------------------
# 8. Configure Hermes
# -------------------------------------------------------------------
CONFIG_FILE="$HOME/.hermes/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    if grep -q "provider: neural" "$CONFIG_FILE" 2>/dev/null; then
        print_ok "Hermes already configured for neural memory"
    else
        print_info "Adding neural memory to config.yaml..."
        # Check if memory section exists
        if grep -q "^memory:" "$CONFIG_FILE" 2>/dev/null; then
            # Update existing memory section
            $PYTHON -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f) or {}
config.setdefault('memory', {})
config['memory']['provider'] = 'neural'
config['memory'].setdefault('neural', {})
config['memory']['neural']['db_path'] = '$DB_PATH'
config['memory']['neural']['embedding_backend'] = 'fastembed'
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('  Updated config.yaml')
" 2>/dev/null && print_ok "Config updated" || print_warn "Manual config needed: add memory.provider: neural"
        else
            print_warn "Add to config.yaml:"
            echo "    memory:"
            echo "      provider: neural"
        fi
    fi
else
    print_warn "Config not found at $CONFIG_FILE"
    echo "  Create config.yaml with:"
    echo "    memory:"
    echo "      provider: neural"
    echo "      neural:"
    echo "        db_path: $DB_PATH"
    echo "        embedding_backend: fastembed"
fi

# -------------------------------------------------------------------
# 9. Verify
# -------------------------------------------------------------------
print_info "Verifying..."

$PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import NeuralMemory
m = NeuralMemory(embedding_backend='hash', use_cpp=False)
print(f'  NeuralMemory: OK ({m.stats()["memories"]} memories)')
m.close()
" 2>/dev/null && print_ok "Verification passed" || print_err "Verification failed"

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${GREEN} Installation complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo ""
echo "  Config: memory.provider: neural"
echo "  Backend: FastEmbed (intfloat/multilingual-e5-large)"
echo "  GPU Recall: $([ -f /usr/bin/nvidia-smi ] && echo 'enabled' || echo 'disabled (no CUDA)')"
echo ""
echo "  Restart hermes: hermes gateway restart"
echo ""
