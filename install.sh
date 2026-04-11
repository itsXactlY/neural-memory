#!/bin/bash
# install.sh — Neural Memory Adapter for Hermes Agent
# Usage: bash install.sh [--lite|--full]
#   --lite   SQLite + hash/tfidf (budget VPS)
#   --full   SQLite + MSSQL + sentence-transformers (production)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
HERMES_DIR="$HOME/.hermes"
NEURAL_DIR="$HOME/.neural_memory"
PLUGIN_DIR="$HERMES_DIR/hermes-agent/plugins/memory/neural"
ENV_FILE="$HERMES_DIR/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
print_info() { echo -e "  ${BLUE}[..]${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}[!!]${NC} $1"; }
print_err()  { echo -e "  ${RED}[XX]${NC} $1"; }

# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Neural Memory Adapter — Hermes Installer"
echo "=============================================="

# Mode selection
MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo ""
    echo "  Select installation mode:"
    echo ""
    echo -e "    ${GREEN}[1]${NC} Lite        — SQLite only, hash/tfidf embeddings"
    echo -e "                     Budget VPS (~50MB RAM, no GPU, no Docker)"
    echo ""
    echo -e "    ${BLUE}[2]${NC} Full Stack  — SQLite + MSSQL + sentence-transformers"
    echo -e "                     Production (~500MB RAM, optional GPU)"
    echo ""
    echo -n "  Choice [1/2]: "
    read -n 1 -r CHOICE
    echo ""

    case "$CHOICE" in
        2) MODE="full" ;;
        *) MODE="lite" ;;
    esac
fi

case "$MODE" in
    --full|full) IS_FULL=true ;;
    *) IS_FULL=false ;;
esac

echo ""
if $IS_FULL; then
    echo -e "  Mode: ${BLUE}Full Stack${NC}"
else
    echo -e "  Mode: ${GREEN}Lite${NC}"
fi

# ---------------------------------------------------------------------------
# [1/6] Prerequisites
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_err "python3 not found"
    exit 1
fi
echo "  python3: $(python3 --version)"

# Core deps (always needed)
python3 -c "import numpy" 2>/dev/null && print_ok "numpy" || {
    print_info "Installing numpy..."
    pip install --quiet numpy 2>/dev/null || pip install --user --quiet numpy
    print_ok "numpy installed"
}

# Cython (for fast_ops acceleration)
python3 -c "import Cython" 2>/dev/null && print_ok "cython" || {
    print_info "Installing cython..."
    pip install --quiet cython 2>/dev/null || pip install --user --quiet cython
    print_ok "cython installed"
}
# Build fast_ops
if [ -f "$PROJECT_DIR/python/setup_fast.py" ]; then
    print_info "Building Cython fast_ops..."
    cd "$PROJECT_DIR/python" && python3 setup_fast.py build_ext --inplace 2>/dev/null && print_ok "fast_ops compiled" || print_warn "fast_ops build failed (optional)"
fi

# sentence-transformers (Full Stack only)
if $IS_FULL; then
    python3 -c "import sentence_transformers" 2>/dev/null && print_ok "sentence-transformers" || {
        print_info "Installing sentence-transformers (~200MB download)..."
        pip install --quiet sentence-transformers 2>/dev/null || pip install --user --quiet sentence-transformers
        print_ok "sentence-transformers installed"
    }

    python3 -c "import pyodbc" 2>/dev/null && print_ok "pyodbc" || {
        print_warn "pyodbc not found. Install with: pip install pyodbc"
    }
fi

# ---------------------------------------------------------------------------
# [2/6] Build C++ library (optional)
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Building C++ library (optional)..."
if command -v cmake &> /dev/null && command -v g++ &> /dev/null; then
    mkdir -p "$PROJECT_DIR/build"
    cd "$PROJECT_DIR/build"

    local CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
    if ! $IS_FULL; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DUSE_MSSQL=OFF"
    fi
    if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
        cmake .. $CMAKE_FLAGS 2>&1 | tail -1
    else
        cmake .. $CMAKE_FLAGS \
            -DCMAKE_CXX_FLAGS="-O3 -march=x86-64" 2>&1 | tail -1
    fi
    cmake --build . -j$(nproc) 2>&1 | tail -1
    print_ok "C++ library built"
else
    print_warn "Skipped (cmake/g++ not found, Python-only mode)"
fi

# ---------------------------------------------------------------------------
# [3/6] Create directories
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Creating directories..."
mkdir -p "$NEURAL_DIR"
mkdir -p "$NEURAL_DIR/models"
print_ok "$NEURAL_DIR"

# ---------------------------------------------------------------------------
# [4/6] Database setup
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Database setup..."
if $IS_FULL; then
    bash "$PROJECT_DIR/install_database.sh" --full
else
    bash "$PROJECT_DIR/install_database.sh" --lite
fi

# ---------------------------------------------------------------------------
# [5/6] Install Hermes plugin
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Installing Hermes plugin..."
mkdir -p "$PLUGIN_DIR"

# Core files (always)
cp "$PROJECT_DIR/hermes-plugin/__init__.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/plugin.yaml" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/memory_client.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/embed_provider.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/neural_memory.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/cpp_bridge.py" "$PLUGIN_DIR/"

# Dream engine files (always)
cp "$PROJECT_DIR/hermes-plugin/dream_engine.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/dream_mssql_store.py" "$PLUGIN_DIR/"

# MSSQL (Full Stack only)
if $IS_FULL; then
    cp "$PROJECT_DIR/hermes-plugin/mssql_store.py" "$PLUGIN_DIR/"
    print_ok "Plugin files (Full Stack: + mssql_store)"
else
    print_ok "Plugin files (Lite)"
fi

# Skin
echo ""
echo "  Installing neural skin..."
SKINS_DIR="$HERMES_DIR/skins"
mkdir -p "$SKINS_DIR"
cp "$PROJECT_DIR/hermes-plugin/neural_skin.yaml" "$SKINS_DIR/neural.yaml" 2>/dev/null && \
    print_ok "Skin: $SKINS_DIR/neural.yaml" || print_warn "Skin file not found, skipping"

# Skills
echo ""
echo "  Installing skills..."
SKILLS_DIR="$HERMES_DIR/skills/devops"
mkdir -p "$SKILLS_DIR"

if [ -d "$PROJECT_DIR/skills/neural-dream-engine" ]; then
    cp "$PROJECT_DIR/skills/neural-dream-engine/SKILL.md" "$SKILLS_DIR/neural-dream-engine/"
    print_ok "Skill: devops/neural-dream-engine"
else
    print_warn "Skill directory not found, skipping"
fi

# ---------------------------------------------------------------------------
# [6/6] Update Hermes config
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Updating Hermes config..."
CONFIG="$HERMES_DIR/config.yaml"

if [ ! -f "$CONFIG" ]; then
    print_warn "$CONFIG not found. Create it first with 'hermes setup'"
    echo "    Then set:"
    echo "      memory:"
    echo "        provider: neural"
    echo "        neural:"
    if $IS_FULL; then
        echo "          db_path: ~/.neural_memory/memory.db"
        echo "          embedding_backend: sentence-transformers"
        echo "          dream:"
        echo "            mssql:"
        echo "              server: localhost"
        echo "              database: NeuralMemory"
    else
        echo "          db_path: ~/.neural_memory/memory.db"
        echo "          embedding_backend: auto"
    fi
else
    # Build config snippet
    if $IS_FULL; then
        NEURAL_CFG="      neural:
        db_path: ~/.neural_memory/memory.db
        embedding_backend: sentence-transformers
        prefetch_limit: 10
        search_limit: 10
        dream:
          mssql:
            server: localhost
            database: NeuralMemory"
    else
        NEURAL_CFG="      neural:
        db_path: ~/.neural_memory/memory.db
        embedding_backend: auto
        prefetch_limit: 5
        search_limit: 5
        dream:
          enabled: true
          idle_threshold: 600"
    fi

    if grep -q "provider: neural" "$CONFIG"; then
        print_ok "Already configured: provider: neural"
    else
        print_info "Adding neural memory config..."
        if grep -q "memory:" "$CONFIG"; then
            # Replace existing provider
            sed -i 's/provider: mempalace/provider: neural/' "$CONFIG" 2>/dev/null || true
            sed -i 's/provider: honcho/provider: neural/' "$CONFIG" 2>/dev/null || true
            sed -i 's/provider: retaindb/provider: neural/' "$CONFIG" 2>/dev/null || true

            if ! grep -q "neural:" "$CONFIG"; then
                sed -i "/provider: neural/a\\
$NEURAL_CFG" "$CONFIG"
            fi
            print_ok "Config updated"
        else
            print_warn "No memory section found. Add manually to $CONFIG:"
            echo "    memory:"
            echo "      provider: neural"
            echo "$NEURAL_CFG"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
if $IS_FULL; then
    echo "  Mode:      Full Stack"
    echo "  SQLite:    ~/.neural_memory/memory.db"
    echo "  MSSQL:     localhost/NeuralMemory"
    echo "  Embedding: sentence-transformers"
    echo "  Dream:     MSSQL + SQLite (3 phases)"
else
    echo "  Mode:      Lite"
    echo "  SQLite:    ~/.neural_memory/memory.db"
    echo "  Embedding: hash/tfidf (auto-detect)"
    echo "  Dream:     SQLite only (3 phases)"
fi
echo ""
echo "  Start:  hermes"
echo "  Test:   cd $PROJECT_DIR/python && python3 demo.py"
echo ""
echo "  Tools:"
echo "    neural_remember      — Store a memory"
echo "    neural_recall        — Search memories"
echo "    neural_think         — Spreading activation"
echo "    neural_graph         — Knowledge graph stats"
echo "    neural_dream         — Force dream cycle"
echo "    neural_dream_stats   — Dream statistics"
echo ""
echo "  Dream Cron: runs every 6h (or idle-based)"
echo ""
