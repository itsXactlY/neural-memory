#!/bin/bash
#
# Neural Memory Adapter — Installer
#
# Installs the neural memory plugin directly into the hermes-agent
# plugin directory. No fork required — works with ANY hermes-agent
# installation (upstream or fork).
#
# Usage:
#   bash install.sh                         # auto-detect hermes-agent
#   bash install.sh /path/to/hermes-agent   # explicit path
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
BENCH_DIR="$SCRIPT_DIR/benchmarks"

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
echo "╔══════════════════════════════════════════════╗"
echo "║   Neural Memory Adapter — Installer          ║"
echo "║   Local semantic memory for hermes-agent     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---------------------------------------------------------------------------
# 1. Detect hermes-agent installation
# ---------------------------------------------------------------------------

HERMES_AGENT=""

# Explicit path
if [ -n "$1" ]; then
    HERMES_AGENT="$1"
fi

# Auto-detect
if [ -z "$HERMES_AGENT" ]; then
    # Check common locations
    for candidate in \
        "$HOME/.hermes/hermes-agent" \
        "$HOME/hermes-agent" \
        "/opt/hermes-agent" \
        "$(which hermes 2>/dev/null && echo '')"; do
        if [ -n "$candidate" ] && [ -d "$candidate/plugins/memory" ]; then
            HERMES_AGENT="$candidate"
            break
        fi
    done
fi

# Check hermes CLI
if [ -z "$HERMES_AGENT" ]; then
    HERMES_BIN=$(which hermes 2>/dev/null || true)
    if [ -n "$HERMES_BIN" ]; then
        # Follow symlinks, find parent
        REAL_BIN=$(readlink -f "$HERMES_BIN" 2>/dev/null || echo "$HERMES_BIN")
        POSSIBLE=$(dirname "$(dirname "$REAL_BIN")")
        if [ -d "$POSSIBLE/plugins/memory" ]; then
            HERMES_AGENT="$POSSIBLE"
        fi
    fi
fi

if [ -z "$HERMES_AGENT" ] || [ ! -d "$HERMES_AGENT/plugins/memory" ]; then
    print_err "hermes-agent not found!"
    echo ""
    echo "  Install hermes-agent first, then run:"
    echo "    bash install.sh /path/to/hermes-agent"
    echo ""
    exit 1
fi

PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"
print_ok "hermes-agent: $HERMES_AGENT"
print_ok "Plugin target: $PLUGIN_DIR"

# ---------------------------------------------------------------------------
# 2. Python check
# ---------------------------------------------------------------------------

PYTHON=${PYTHON:-python3}
if ! $PYTHON --version &>/dev/null; then
    print_err "Python 3 not found"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1)
print_ok "Python: $PY_VER"

# ---------------------------------------------------------------------------
# 3. Dependencies
# ---------------------------------------------------------------------------

print_info "Checking dependencies..."

# pyodbc (for MSSQL, optional)
$PYTHON -c "import pyodbc" 2>/dev/null && print_ok "pyodbc (MSSQL)" || print_warn "pyodbc not found — MSSQL backend unavailable (optional)"

# numpy
$PYTHON -c "import numpy" 2>/dev/null && print_ok "numpy" || {
    print_info "Installing numpy..."
    pip install --quiet numpy 2>/dev/null || pip install --user --quiet numpy
    print_ok "numpy installed"
}

# Cython (for fast_ops)
$PYTHON -c "import Cython" 2>/dev/null && print_ok "Cython" || {
    print_info "Installing Cython..."
    pip install --quiet cython 2>/dev/null || pip install --user --quiet cython
    print_ok "Cython installed"
}

# sentence-transformers (optional, for better embeddings)
$PYTHON -c "import sentence_transformers" 2>/dev/null && print_ok "sentence-transformers" || print_warn "sentence-transformers not found — using hash/tfidf backends (optional)"

# networkx (optional, enables Louvain community detection in dream Insight phase)
$PYTHON -c "import networkx" 2>/dev/null && print_ok "networkx (Louvain)" || print_warn "networkx not found — Insight phase falls back to BFS connected-components (optional)"

# hnswlib (optional, enables scalable ANN retrieval in Python-only mode)
$PYTHON -c "import hnswlib" 2>/dev/null && print_ok "hnswlib (HNSW index)" || print_warn "hnswlib not found — recall uses brute-force cosine when C++ unavailable (optional)"

# ---------------------------------------------------------------------------
# 4. Create plugin directory
# ---------------------------------------------------------------------------

print_info "Installing plugin..."
mkdir -p "$PLUGIN_DIR"

# Core files
for f in __init__.py plugin.yaml memory_client.py embed_provider.py neural_memory.py \
         cpp_bridge.py mssql_store.py dream_mssql_store.py dream_engine.py README.md; do
    if [ -f "$PYTHON_DIR/$f" ]; then
        cp "$PYTHON_DIR/$f" "$PLUGIN_DIR/"
    fi
done

# fast_ops source
cp "$PYTHON_DIR/fast_ops.pyx" "$PLUGIN_DIR/" 2>/dev/null || true

print_ok "Plugin files installed"

# ---------------------------------------------------------------------------
# 5. Build Cython fast_ops
# ---------------------------------------------------------------------------

print_info "Building Cython fast_ops..."
if [ -f "$PYTHON_DIR/setup_fast.py" ]; then
    cd "$PYTHON_DIR"
    if $PYTHON setup_fast.py build_ext --inplace 2>/dev/null; then
        SO_FILE=$(ls "$PYTHON_DIR"/fast_ops.cpython*.so 2>/dev/null | head -1)
        if [ -n "$SO_FILE" ]; then
            cp "$SO_FILE" "$PLUGIN_DIR/"
            print_ok "fast_ops compiled and installed"
        fi
    else
        print_warn "fast_ops build failed — Python fallback active (cosine_similarity slower but works)"
    fi
else
    print_warn "setup_fast.py not found — skipping Cython build"
fi

# ---------------------------------------------------------------------------
# 6. Build C++ library (optional)
# ---------------------------------------------------------------------------

print_info "Checking C++ library..."
CPP_LIB="$SCRIPT_DIR/build/libneural_memory.so"
if [ -f "$CPP_LIB" ]; then
    print_ok "C++ bridge available: $CPP_LIB"
else
    if [ -d "$SCRIPT_DIR/build" ] && [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
        print_info "Building C++ library..."
        cd "$SCRIPT_DIR/build"
        cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MSSQL=OFF 2>/dev/null && \
        make neural_memory -j$(nproc) 2>/dev/null && \
        print_ok "C++ bridge built" || \
        print_warn "C++ build failed — Python fallback active"
    else
        print_warn "C++ source not found — Python fallback active"
    fi
fi

# ---------------------------------------------------------------------------
# 7. Initialize database
# ---------------------------------------------------------------------------

DB_PATH="${NEURAL_MEMORY_DB_PATH:-$HOME/.neural_memory/memory.db}"
if [ ! -f "$DB_PATH" ]; then
    print_info "Initializing database at $DB_PATH..."
    mkdir -p "$(dirname "$DB_PATH")"
    $PYTHON -c "
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
print(f'  memories: {s.stats()[\"memories\"]}')
s.close()
" 2>/dev/null && print_ok "Database initialized" || print_warn "Database init failed (will auto-create on first use)"
else
    $PYTHON -c "
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
stats = s.stats()
s.close()
print(f'  {stats[\"memories\"]} memories, {stats[\"connections\"]} connections')
" 2>/dev/null && print_ok "Existing database found" || print_warn "Database exists but may be corrupted"
fi

# ---------------------------------------------------------------------------
# 8. Verify installation
# ---------------------------------------------------------------------------

print_info "Verifying installation..."

$PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from plugins.memory import discover_memory_providers
providers = {n: (d, a) for n, d, a in discover_memory_providers()}
if 'neural' in providers:
    desc, avail = providers['neural']
    status = 'available' if avail else 'not available (check deps)'
    print(f'  neural: {status}')
else:
    print('  neural: NOT FOUND in plugin directory')
    sys.exit(1)
" 2>/dev/null && print_ok "Plugin discoverable" || {
    # Fallback: test import directly
    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import NeuralMemory
m = NeuralMemory(embedding_backend='hash', use_cpp=False)
print(f'  NeuralMemory: {m.stats()[\"memories\"]} memories')
m.close()
" 2>/dev/null && print_ok "NeuralMemory importable" || print_err "Verification failed"
}

# ---------------------------------------------------------------------------
# 9. Configuration hint
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${GREEN} Installation complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo ""
echo "  To activate, add to config.yaml:"
echo ""
echo "    memory:"
echo "      provider: neural"
echo ""
echo "  Or run: hermes memory setup"
echo "          → select 'neural' from the list"
echo ""
echo "  Optional: MSSQL backend"
echo "    Add to config.yaml:"
echo "      memory:"
echo "        neural:"
echo "          dream:"
echo "            mssql:"
echo "              server: 127.0.0.1"
echo "              database: NeuralMemory"
echo "              username: SA"
echo "              password: <your-password>"
echo ""
echo "  Restart hermes to load the plugin."
echo ""
