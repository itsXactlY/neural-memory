#!/bin/bash
# install.sh - Neural Memory Adapter for Hermes Agent
# Usage: bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
HERMES_DIR="$HOME/.hermes"
NEURAL_DIR="$HOME/.neural_memory"
PLUGIN_DIR="$HERMES_DIR/hermes-agent/plugins/memory/neural"

echo "=============================================="
echo "  Neural Memory Adapter - Hermes Installer"
echo "=============================================="

# 1. Prerequisites
echo ""
echo "[1/5] Checking prerequisites..."

# Python
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: python3 not found"
    exit 1
fi
echo "  python3: $(python3 --version)"

# pip dependencies
echo "  Installing Python dependencies..."
pip install --quiet sentence-transformers numpy 2>/dev/null
echo "  sentence-transformers: OK"
echo "  numpy: OK"

# 2. Build C++ library (optional)
echo ""
echo "[2/5] Building C++ library (optional)..."
if command -v cmake &> /dev/null && command -v g++ &> /dev/null; then
    mkdir -p "$PROJECT_DIR/build"
    cd "$PROJECT_DIR/build"
    
    # Apply AVX2 flags (production) or x86-64 (sandbox)
    if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
        echo "  CPU: AVX2 detected"
        cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
    else
        echo "  CPU: No AVX2, using scalar fallback"
        cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS="-O3 -march=x86-64" 2>&1 | tail -1
    fi
    
    cmake --build . -j$(nproc) 2>&1 | tail -1
    echo "  C++ library: OK"
else
    echo "  Skipped (cmake/g++ not found, Python-only mode)"
fi

# 3. Create directories
echo ""
echo "[3/5] Creating directories..."
mkdir -p "$NEURAL_DIR"
mkdir -p "$NEURAL_DIR/models"
echo "  $NEURAL_DIR: OK"

# 4. Install Hermes plugin
echo ""
echo "[4/6] Installing Hermes plugin..."
mkdir -p "$PLUGIN_DIR"

# Copy plugin files
cp "$PROJECT_DIR/hermes-plugin/__init__.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/config.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/plugin.yaml" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/memory_client.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/hermes-plugin/embed_provider.py" "$PLUGIN_DIR/"

# Install .pth for banner monkey-patch (runs before banner display)
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp "$PROJECT_DIR/hermes-plugin/neural_memory_banner.pth" "$SITE_PKG/"
echo "  Banner patch: $SITE_PKG/neural_memory_banner.pth"

echo "  Plugin files: OK"

# 5. Install skin
echo ""
echo "[5/6] Installing neural skin..."
SKINS_DIR="$HERMES_DIR/skins"
mkdir -p "$SKINS_DIR"
cp "$PROJECT_DIR/hermes-plugin/neural_skin.yaml" "$SKINS_DIR/neural.yaml"
echo "  Skin: $SKINS_DIR/neural.yaml"

# 6. Update Hermes config
echo ""
echo "[5/5] Updating Hermes config..."
CONFIG="$HERMES_DIR/config.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "  WARNING: $CONFIG not found - create it first with 'hermes setup'"
    echo "  Then add manually:"
    echo "    memory:"
    echo "      provider: neural"
    echo "      neural:"
    echo "        db_path: ~/.neural_memory/hermes.db"
    echo "        embedding_backend: auto"
    exit 0
fi

# Check if already configured
if grep -q "provider: neural" "$CONFIG"; then
    echo "  Already configured: provider: neural"
else
    # Check current provider
    CURRENT=$(grep "provider:" "$CONFIG" | grep -A0 "memory:" | head -1 || echo "")
    
    if [ -n "$CURRENT" ]; then
        echo "  Current memory provider found."
        echo ""
        read -p "  Switch to neural? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Replace provider line
            sed -i 's/provider: mempalace/provider: neural/' "$CONFIG" 2>/dev/null || true
            sed -i 's/provider: honcho/provider: neural/' "$CONFIG" 2>/dev/null || true
            sed -i 's/provider: mem0/provider: neural/' "$CONFIG" 2>/dev/null || true
            
            # Add neural config section if missing
            if ! grep -q "neural:" "$CONFIG"; then
                # Find the line with "provider: neural" and add config after it
                sed -i '/provider: neural/a\  neural:\n    db_path: ~/.neural_memory/hermes.db\n    embedding_backend: auto\n    consolidation_interval: 300\n    max_episodic: 50000' "$CONFIG"
            fi
            echo "  Config updated!"
        else
            echo "  Skipped. To enable manually, set:"
            echo "    memory:"
            echo "      provider: neural"
        fi
    else
        echo "  No memory provider found. Add to $CONFIG:"
        echo "    memory:"
        echo "      provider: neural"
        echo "      neural:"
        echo "        db_path: ~/.neural_memory/hermes.db"
        echo "        embedding_backend: auto"
    fi
fi

# Done
echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "  Start: hermes"
echo "  Test:  cd $PROJECT_DIR/python && python3 demo.py"
echo ""
echo "  Tools available:"
echo "    neural_remember  - Store a memory"
echo "    neural_recall    - Search memories"
echo "    neural_think     - Spreading activation"
echo "    neural_graph     - Knowledge graph stats"
echo ""
echo "  Model cache: ~/.neural_memory/models/ (~87MB, once)"
echo "  Memory DB:   ~/.neural_memory/hermes.db"
echo ""
