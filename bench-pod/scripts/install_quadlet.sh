#!/usr/bin/env bash
# Install the Comparison Pod Quadlet manifest into
# ~/.config/containers/systemd/bench/ and reload systemd --user.
#
# Idempotent — copying over an existing manifest is fine. The unit is
# `bench-pod.service` (Quadlet generates this from `bench.pod`).

set -euo pipefail

POD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/.config/containers/systemd/bench"

log() { printf '[install-quadlet] %s\n' "$*"; }

mkdir -p "$DEST"
cp -f "$POD_ROOT/quadlet/bench.pod" "$DEST/bench.pod"
log "installed: $DEST/bench.pod"

if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not found; skip daemon-reload"
    exit 0
fi

if ! systemctl --user list-units >/dev/null 2>&1; then
    log "systemd --user not reachable; skip daemon-reload"
    exit 0
fi

systemctl --user daemon-reload
log "systemd --user daemon-reload OK"

if systemctl --user list-unit-files | grep -q '^bench-pod\.service'; then
    log "unit available: bench-pod.service"
    log "start with:    systemctl --user start bench-pod.service"
    log "status with:   systemctl --user status bench-pod.service"
else
    log "bench-pod.service not yet generated (Quadlet may need a podman version refresh)"
fi
