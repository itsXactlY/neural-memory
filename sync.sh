#!/bin/bash
# Sync python/ (source of truth) to hermes-agent plugin directory
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/python"

# Auto-detect hermes-agent
HERMES_AGENT="${1:-$HOME/.hermes/hermes-agent}"
PLUGIN="$HERMES_AGENT/plugins/memory/neural"

if [ ! -d "$PLUGIN" ]; then
    echo "Plugin dir not found: $PLUGIN"
    echo "Usage: $0 [/path/to/hermes-agent]"
    exit 1
fi

echo "Syncing: python/ → $PLUGIN"

for f in memory_client.py cpp_bridge.py embed_provider.py neural_memory.py \
         mssql_store.py dream_mssql_store.py dream_engine.py fast_ops.pyx; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$PLUGIN/$f"
        echo "  $f"
    fi
done

# fast_ops .so
if ls "$SRC"/fast_ops.cpython*.so 1>/dev/null 2>&1; then
    cp "$SRC"/fast_ops.cpython*.so "$PLUGIN/"
    echo "  fast_ops.so"
fi

echo "Done."
