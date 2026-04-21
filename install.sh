#!/bin/bash
#
# Neural Memory Adapter — Installer v2 (2026-04-21)
#
# Architecture:
#   python/           — SINGLE SOURCE OF TRUTH (all .py files live here)
#   hermes-plugin/    — Hermes-specific metadata only (plugin.yaml, neural_skin.yaml)
#   Deploy targets    — SYMLINKS back to python/ (not copies)
#
# Usage:
#   bash install.sh install [OPTIONS] [/path/to/hermes-agent]
#   bash install.sh update
#   bash install.sh test
#   bash install.sh verify
#   bash install.sh uninstall
#
# Options (install only):
#   --hash-backend   Use hash embedding (instant, no model download, low RAM)
#
set -e

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
PLUGIN_SRC="$SCRIPT_DIR/hermes-plugin"

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

# -------------------------------------------------------------------
# Root check
# -------------------------------------------------------------------
check_not_root() {
    if [ "$(id -u)" -eq 0 ]; then
        echo -e "${RED}DO NOT RUN AS ROOT.${NC}"
        echo "This installer writes to ~/.hermes/ and uses the user's venv."
        echo "Run as your normal user."
        exit 1
    fi
}

# -------------------------------------------------------------------
# Banner
# -------------------------------------------------------------------
print_banner() {
    echo -e "${BOLD}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║   Neural Memory Adapter — Installer v2           ║"
    echo "║   Symlinks · FastEmbed + GPU · SQLite-First      ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# -------------------------------------------------------------------
# Detect hermes-agent
# -------------------------------------------------------------------
detect_hermes_agent() {
    local ARG="$1"
    local HERMES_AGENT=""

    if [ -n "$ARG" ]; then
        HERMES_AGENT="$ARG"
    fi

    if [ -z "$HERMES_AGENT" ]; then
    for candidate in \
        "$HOME/.hermes/hermes-agent" \
        "$HOME/jack-in-a-box/hermes-agent" \
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
        echo "    bash install.sh install /path/to/hermes-agent"
        exit 1
    fi

    echo "$HERMES_AGENT"
}

# -------------------------------------------------------------------
# Detect Python + pip
# -------------------------------------------------------------------
detect_python() {
    PYTHON=${PYTHON:-python3}
    if ! $PYTHON --version &>/dev/null; then
        print_err "Python 3 not found"
        exit 1
    fi
    PY_VER=$($PYTHON --version 2>&1)
    print_ok "Python: $PY_VER"
}

detect_pip() {
    local HERMES_AGENT="$1"

    PIP=""
    PIP_ARGS=""

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
    fi
}

# -------------------------------------------------------------------
# Create symlinks for a target directory
# -------------------------------------------------------------------
create_symlinks() {
    local TARGET_DIR="$1"
    local CREATED=0

    mkdir -p "$TARGET_DIR"

    # Symlink all .py files from python/
    for f in "$PYTHON_DIR"/*.py; do
        [ -f "$f" ] || continue
        local BASENAME
        BASENAME=$(basename "$f")
        local LINK="$TARGET_DIR/$BASENAME"

        # Backup existing regular file (not symlink)
        if [ -f "$LINK" ] && [ ! -L "$LINK" ]; then
            local BACKUP="${LINK}.bak.$(date +%Y%m%d_%H%M%S)"
            mv "$LINK" "$BACKUP"
            echo "    Backed up: $BASENAME → $(basename "$BACKUP")" >&2
        fi

        # Remove stale symlink
        rm -f "$LINK"

        # Create symlink
        ln -s "$f" "$LINK"
        CREATED=$((CREATED + 1))
    done

    # Symlink optional compiled files
    for f in "$PYTHON_DIR"/fast_ops.cpython*.so "$PYTHON_DIR"/fast_ops.pyx; do
        [ -f "$f" ] || continue
        local BASENAME
        BASENAME=$(basename "$f")
        local LINK="$TARGET_DIR/$BASENAME"

        if [ -f "$LINK" ] && [ ! -L "$LINK" ]; then
            local BACKUP="${LINK}.bak.$(date +%Y%m%d_%H%M%S)"
            mv "$LINK" "$BACKUP"
            echo "    Backed up: $BASENAME → $(basename "$BACKUP")" >&2
        fi

        rm -f "$LINK"
        ln -s "$f" "$LINK"
        CREATED=$((CREATED + 1))
    done

    # Copy hermes-specific files (NOT symlinked — they are per-install metadata)
    for f in plugin.yaml neural_skin.yaml; do
        if [ -f "$PLUGIN_SRC/$f" ]; then
            cp "$PLUGIN_SRC/$f" "$TARGET_DIR/"
        fi
    done

    echo "$CREATED"
}

# -------------------------------------------------------------------
# Remove symlinks from a target directory
# -------------------------------------------------------------------
remove_symlinks() {
    local TARGET_DIR="$1"
    local REMOVED=0

    if [ ! -d "$TARGET_DIR" ]; then
        return 0
    fi

    for f in "$TARGET_DIR"/*; do
        [ -e "$f" ] || continue
        local BASENAME
        BASENAME=$(basename "$f")

        if [ -L "$f" ]; then
            # Symlink — remove it
            rm -f "$f"
            REMOVED=$((REMOVED + 1))
        elif [[ "$BASENAME" == *.bak.* ]]; then
            # Backup file — leave it (user might want to restore)
            :
        fi
    done

    # Remove hermes-specific files we copied
    rm -f "$TARGET_DIR/plugin.yaml" "$TARGET_DIR/neural_skin.yaml" 2>/dev/null || true

    # Remove __pycache__
    rm -rf "$TARGET_DIR/__pycache__" 2>/dev/null || true

    echo "$REMOVED"
}

# -------------------------------------------------------------------
# COMMAND: install
# -------------------------------------------------------------------
cmd_install() {
    check_not_root
    print_banner

    # Parse args
    local HASH_BACKEND=false
    local WITH_MSSQL=false
    local HERMES_AGENT_ARG=""

    for arg in "$@"; do
        case "$arg" in
            --hash-backend) HASH_BACKEND=true ;;
            --with-mssql) WITH_MSSQL=true ;;
            --help|-h)
                echo "Usage: bash install.sh install [OPTIONS] [/path/to/hermes-agent]"
                echo ""
                echo "Options:"
                echo "  --hash-backend   Use hash embedding (instant, no model download)"
                echo "  --with-mssql     Also set up MSSQL cold store (requires pyodbc)"
                exit 0
                ;;
            *) HERMES_AGENT_ARG="$arg" ;;
        esac
    done

    # Detect hermes-agent
    local HERMES_AGENT
    HERMES_AGENT=$(detect_hermes_agent "$HERMES_AGENT_ARG")
    local PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"
    local ALT_PLUGIN_DIR="$HOME/.hermes/plugins/memory/neural"

    print_ok "hermes-agent: $HERMES_AGENT"
    print_ok "Plugin target: $PLUGIN_DIR"
    print_ok "Alt target:    $ALT_PLUGIN_DIR"

    # Detect Python + pip
    detect_python
    detect_pip "$HERMES_AGENT"

    # -------------------------------------------------------------------
    # System checks
    # -------------------------------------------------------------------
    TOTAL_RAM_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
    TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "0")
    print_ok "RAM: ${TOTAL_RAM_GB}GB"

    if [ "$TOTAL_RAM_MB" -lt 3072 ]; then
        print_warn "Less than 3GB RAM — auto-selecting --hash-backend"
        HASH_BACKEND=true
    fi

    # -------------------------------------------------------------------
    # Dependencies
    # -------------------------------------------------------------------
    print_info "Upgrading pip..."
    $PIP install --quiet --upgrade pip 2>/dev/null || true

    print_info "Installing core dependencies..."

    # numpy FIRST
    $PYTHON -c "import numpy" 2>/dev/null && print_ok "numpy" || {
        print_info "Installing numpy..."
        $PIP install $PIP_ARGS --quiet numpy
        print_ok "numpy installed"
    }

    # FastEmbed (unless hash backend)
    if [ "$HASH_BACKEND" = true ]; then
        print_info "Skipping FastEmbed (--hash-backend mode)"
        print_ok "Hash backend: instant, zero deps, 1024d"
    else
        $PYTHON -c "import fastembed" 2>/dev/null && print_ok "fastembed" || {
            print_info "Installing fastembed..."
            $PIP install $PIP_ARGS --quiet fastembed
            print_ok "fastembed installed"
        }
        $PYTHON -c "from fastembed import TextEmbedding" 2>/dev/null && print_ok "FastEmbed import OK" || print_warn "FastEmbed import failed"
    fi

    # Optional: GPU (torch + CUDA)
    echo ""
    HAS_CUDA=false
    if command -v nvidia-smi &>/dev/null; then
        $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && {
            HAS_CUDA=true
            print_ok "torch + CUDA detected"
        } || {
            print_info "NVIDIA GPU found — installing torch with CUDA..."
            CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
            if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 12 ]; then
                $PIP install $PIP_ARGS --quiet torch --index-url https://download.pytorch.org/whl/cu121
            elif [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 11 ]; then
                $PIP install $PIP_ARGS --quiet torch --index-url https://download.pytorch.org/whl/cu118
            else
                $PIP install $PIP_ARGS --quiet torch --index-url https://download.pytorch.org/whl/cpu
            fi
            $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && HAS_CUDA=true || print_warn "CUDA not available after install"
        }
    fi

    # Optional: sentence-transformers, pyodbc
    $PYTHON -c "import sentence_transformers" 2>/dev/null && print_ok "sentence-transformers" || print_info "sentence-transformers: optional, install later"
    $PYTHON -c "import pyodbc" 2>/dev/null && print_ok "pyodbc" || print_info "pyodbc: optional (MSSQL mirror)"

    # -------------------------------------------------------------------
    # Create symlinks
    # -------------------------------------------------------------------
    echo ""
    print_info "Creating symlinks from $PYTHON_DIR..."

    local COUNT1
    COUNT1=$(create_symlinks "$PLUGIN_DIR")
    print_ok "Primary: $PLUGIN_DIR ($COUNT1 symlinks)"

    local COUNT2
    COUNT2=$(create_symlinks "$ALT_PLUGIN_DIR")
    print_ok "Alt:     $ALT_PLUGIN_DIR ($COUNT2 symlinks)"

    # -------------------------------------------------------------------
    # Initialize database
    # -------------------------------------------------------------------
    DB_PATH="${NEURAL_MEMORY_DB_PATH:-$HOME/.neural_memory/memory.db}"
    mkdir -p "$(dirname "$DB_PATH")"

    if [ ! -f "$DB_PATH" ]; then
        print_info "Creating database at $DB_PATH..."
        $PYTHON -c "
import sys; sys.path.insert(0, '$PLUGIN_DIR')
from memory_client import SQLiteStore
s = SQLiteStore('$DB_PATH')
s.close()
print('  Created')
" 2>/dev/null && print_ok "Database initialized" || print_warn "Database will auto-create on first use"
    else
        print_ok "Existing database found"
    fi

    # -------------------------------------------------------------------
    # Optional: MSSQL cold store setup
    # -------------------------------------------------------------------
    if [ "$WITH_MSSQL" = true ]; then
        echo ""
        print_info "Setting up MSSQL cold store (--with-mssql)..."
        if [ -f "$SCRIPT_DIR/install_database.sh" ]; then
            bash "$SCRIPT_DIR/install_database.sh" install --full 2>/dev/null && \
                print_ok "MSSQL cold store configured" || \
                print_warn "MSSQL setup had issues (check install_database.sh install --full)"
        else
            print_warn "install_database.sh not found — skipping MSSQL setup"
        fi
    fi

    # Embedding backend for config
    if [ "$HASH_BACKEND" = true ]; then
        EMBED_BACKEND="hash"
    else
        EMBED_BACKEND="fastembed"
    fi

    # -------------------------------------------------------------------
    # Configure hermes config.yaml
    # -------------------------------------------------------------------
    CONFIG_FILE="$HOME/.hermes/config.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        if grep -q "provider: neural" "$CONFIG_FILE" 2>/dev/null; then
            print_ok "config.yaml: neural provider configured"
        else
            print_info "Updating config.yaml..."
            $PYTHON -c "
import yaml, os
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f) or {}
config.setdefault('memory', {})
config['memory']['provider'] = 'neural'
config['memory'].setdefault('neural', {})
config['memory']['neural']['db_path'] = '$DB_PATH'
config['memory']['neural']['embedding_backend'] = '$EMBED_BACKEND'
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
" 2>/dev/null && print_ok "config.yaml updated" || print_warn "Could not auto-update config.yaml"
        fi
    fi

    # -------------------------------------------------------------------
    # Verify
    # -------------------------------------------------------------------
    echo ""
    cmd_verify_quiet "$PLUGIN_DIR"

    # -------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Neural Memory Adapter installed!${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
    echo ""
    echo "  Source:     $PYTHON_DIR (symlinked, not copied)"
    echo "  Database:   $DB_PATH (SQLite)"
    echo "  GPU Recall: $([ "$HAS_CUDA" = true ] && echo 'enabled (CUDA)' || echo 'disabled (CPU)')"
    echo "  Symlinks:   $COUNT1 files → $PLUGIN_DIR"
    echo ""
    echo "  Next: hermes gateway restart"
    echo ""
}

# -------------------------------------------------------------------
# COMMAND: update
# -------------------------------------------------------------------
cmd_update() {
    check_not_root
    print_banner

    print_info "Updating neural-memory-adapter..."

    # Git pull
    if [ -d "$SCRIPT_DIR/.git" ]; then
        print_info "Pulling latest changes..."
        cd "$SCRIPT_DIR"
        git pull --ff-only 2>/dev/null && print_ok "Git updated" || print_warn "Git pull skipped (no remote or conflicts)"
    fi

    # Re-detect hermes-agent
    local HERMES_AGENT
    HERMES_AGENT=$(detect_hermes_agent "")
    local PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"
    local ALT_PLUGIN_DIR="$HOME/.hermes/plugins/memory/neural"

    # Remove old symlinks and re-create
    print_info "Refreshing symlinks..."
    remove_symlinks "$PLUGIN_DIR" >/dev/null
    remove_symlinks "$ALT_PLUGIN_DIR" >/dev/null

    local COUNT1
    COUNT1=$(create_symlinks "$PLUGIN_DIR")
    local COUNT2
    COUNT2=$(create_symlinks "$ALT_PLUGIN_DIR")

    print_ok "Symlinks refreshed: $PLUGIN_DIR ($COUNT1), $ALT_PLUGIN_DIR ($COUNT2)"

    # Re-install deps if needed
    detect_python
    detect_pip "$HERMES_AGENT"

    $PYTHON -c "import numpy" 2>/dev/null || $PIP install $PIP_ARGS --quiet numpy
    $PYTHON -c "import fastembed" 2>/dev/null || $PIP install $PIP_ARGS --quiet fastembed

    print_ok "Dependencies verified"

    # Run tests
    echo ""
    cmd_test
}

# -------------------------------------------------------------------
# COMMAND: test
# -------------------------------------------------------------------
cmd_test() {
    check_not_root

    local HERMES_AGENT
    HERMES_AGENT=$(detect_hermes_agent "")

    # Prefer hermes-agent venv python
    if [ -f "$HERMES_AGENT/venv/bin/python3" ]; then
        PYTHON="$HERMES_AGENT/venv/bin/python3"
    fi

    local PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"

    print_info "Running test suite..."

    # Run the bundled test_suite.py via the plugin dir
    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from test_suite import run_all_tests
run_all_tests()
" 2>/dev/null && print_ok "Test suite passed" || {
        print_warn "Test suite had issues — running inline verification..."
        cmd_verify_quiet "$PLUGIN_DIR"
    }
}

# -------------------------------------------------------------------
# COMMAND: verify
# -------------------------------------------------------------------
cmd_verify() {
    check_not_root
    print_banner

    local HERMES_AGENT
    HERMES_AGENT=$(detect_hermes_agent "")
    local PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"

    cmd_verify_quiet "$PLUGIN_DIR"

    # Also verify symlinks are correct
    echo ""
    print_info "Symlink audit:"
    local SYMLINK_COUNT=0
    local BROKEN=0
    for f in "$PLUGIN_DIR"/*.py; do
        [ -L "$f" ] || continue
        SYMLINK_COUNT=$((SYMLINK_COUNT + 1))
        if [ ! -e "$(readlink -f "$f")" ]; then
            print_err "Broken symlink: $(basename "$f")"
            BROKEN=$((BROKEN + 1))
        fi
    done
    print_ok "$SYMLINK_COUNT symlinks, $BROKEN broken"

    # Check source of truth
    echo ""
    print_info "Source of truth: $PYTHON_DIR"
    local SRC_COUNT
    SRC_COUNT=$(ls "$PYTHON_DIR"/*.py 2>/dev/null | wc -l)
    print_ok "$SRC_COUNT .py files in python/"
}

cmd_verify_quiet() {
    local PLUGIN_DIR="$1"

    detect_python

    # Try to find hermes-agent venv python
    local HERMES_AGENT
    HERMES_AGENT=$(detect_hermes_agent "") 2>/dev/null
    if [ -n "$HERMES_AGENT" ] && [ -f "$HERMES_AGENT/venv/bin/python3" ]; then
        PYTHON="$HERMES_AGENT/venv/bin/python3"
    fi

    print_info "Verifying installation..."

    $PYTHON -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')

# Test imports
from memory_client import NeuralMemory
from embed_provider import EmbeddingProvider, HashBackend
print('  Imports: OK')

# Test embedding (hash backend — always available)
hb = HashBackend()
emb = hb.embed('test memory content')
assert len(emb) == 1024, f'Expected 1024d, got {len(emb)}d'
print(f'  Embedding: OK ({len(emb)}d)')

# Test memory system
m = NeuralMemory(embedding_backend='hash', use_cpp=False)
m.remember('installation test', label='test')
results = m.recall('installation test')
assert len(results) > 0, 'Recall returned 0 results'
print(f'  Store+Recall: OK ({len(results)} results)')
m.close()

# Test EmbeddingProvider
try:
    ep = EmbeddingProvider(backend='hash')
    v = ep.embed('hello world')
    assert len(v) == 1024
    print(f'  EmbeddingProvider: OK ({len(v)}d)')
except Exception as e:
    print(f'  EmbeddingProvider: {e}')

# Test FastEmbed if available
try:
    from fastembed import TextEmbedding
    print('  FastEmbed: available')
except ImportError:
    print('  FastEmbed: not installed (optional)')

print('  All checks passed')
" 2>/dev/null && print_ok "Verification passed" || print_warn "Verification had warnings"
}

# -------------------------------------------------------------------
# COMMAND: uninstall
# -------------------------------------------------------------------
cmd_uninstall() {
    check_not_root
    print_banner

    print_info "Uninstalling neural-memory-adapter symlinks..."

    local HERMES_AGENT
    HERMES_AGENT=$(detect_hermes_agent "") 2>/dev/null

    local TOTAL=0

    # Primary target
    if [ -n "$HERMES_AGENT" ]; then
        local PLUGIN_DIR="$HERMES_AGENT/plugins/memory/neural"
        local R1
        R1=$(remove_symlinks "$PLUGIN_DIR")
        print_ok "Removed $R1 symlinks from $PLUGIN_DIR"
        TOTAL=$((TOTAL + R1))

        # Remove directory if empty (excluding .bak files)
        local REMAINING
        REMAINING=$(find "$PLUGIN_DIR" -maxdepth 1 -type f ! -name '*.bak.*' 2>/dev/null | wc -l)
        if [ "$REMAINING" -eq 0 ]; then
            print_info "Directory $PLUGIN_DIR is empty (backups preserved)"
        fi
    fi

    # Alt target
    local ALT_PLUGIN_DIR="$HOME/.hermes/plugins/memory/neural"
    if [ -d "$ALT_PLUGIN_DIR" ]; then
        local R2
        R2=$(remove_symlinks "$ALT_PLUGIN_DIR")
        print_ok "Removed $R2 symlinks from $ALT_PLUGIN_DIR"
        TOTAL=$((TOTAL + R2))
    fi

    echo ""
    print_ok "Uninstalled $TOTAL symlinks total"
    echo ""
    echo "  Source files in $PYTHON_DIR are preserved."
    echo "  Backup files (*.bak.*) are preserved."
    echo "  To fully remove: rm -rf ~/projects/neural-memory-adapter"
    echo "  To remove from config: edit ~/.hermes/config.yaml"
    echo ""
}

# -------------------------------------------------------------------
# Main dispatch
# -------------------------------------------------------------------
CMD="${1:-install}"
shift 2>/dev/null || true

case "$CMD" in
    install)  cmd_install "$@" ;;
    update)   cmd_update "$@" ;;
    test)     cmd_test "$@" ;;
    verify)   cmd_verify "$@" ;;
    uninstall) cmd_uninstall "$@" ;;
    --help|-h)
        echo "Neural Memory Adapter — Installer v2"
        echo ""
        echo "Usage: bash install.sh <command> [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  install    Install symlinks + dependencies (default)"
        echo "  update     Git pull + re-symlink + test"
        echo "  test       Run test suite"
        echo "  verify     Check installation integrity"
        echo "  uninstall  Remove symlinks (preserves source + backups)"
        echo ""
        echo "Install Options:"
        echo "  --hash-backend   Use hash embedding (no model download)"
        echo "  --with-mssql     Also set up MSSQL cold store"
        echo "  /path/to/hermes-agent   Explicit hermes-agent path"
        ;;
    *)
        print_err "Unknown command: $CMD"
        echo "Run: bash install.sh --help"
        exit 1
        ;;
esac
