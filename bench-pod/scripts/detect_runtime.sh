#!/usr/bin/env bash
# Comparison Pod runtime preflight.
#
# Checks (fail-fast):
#   - podman present, version >= 4.4
#   - python3 present
#   - free disk in $HOME at least 20 GB
#   - total RAM at least 8 GB (we recommend 16 GB but allow 8 GB)
#   - docker absent OR if present, warn ("we don't use it, but it's on $PATH")
#
# Exit codes:
#   0  all good
#   1  one or more hard requirements failed

set -euo pipefail

ok()   { printf '[preflight] [ OK ] %s\n' "$*"; }
warn() { printf '[preflight] [WARN] %s\n' "$*"; }
fail() { printf '[preflight] [FAIL] %s\n' "$*"; FAILED=1; }

FAILED=0

# --- podman ---
if ! command -v podman >/dev/null 2>&1; then
    fail "podman not found on PATH. Install Podman 4.4+ (https://podman.io)."
else
    PODMAN_VER="$(podman --version | awk '{print $3}')"
    # Compare semver: major.minor must be >= 4.4
    if [ -z "$PODMAN_VER" ]; then
        fail "could not parse podman version"
    else
        MAJOR="${PODMAN_VER%%.*}"
        REST="${PODMAN_VER#*.}"
        MINOR="${REST%%.*}"
        if [ "$MAJOR" -lt 4 ] || { [ "$MAJOR" -eq 4 ] && [ "$MINOR" -lt 4 ]; }; then
            fail "podman $PODMAN_VER < 4.4 — Quadlet support requires >= 4.4"
        else
            ok "podman $PODMAN_VER"
        fi
    fi
fi

# --- python3 ---
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found"
else
    PYVER="$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
    ok "python3 $PYVER"
fi

# --- disk space ---
# df -BG gives gigabytes; the 4th column is "Available"
AVAIL_GB="$(df -BG "$HOME" | awk 'NR==2 {sub("G","",$4); print $4}')"
if [ -z "${AVAIL_GB:-}" ]; then
    warn "could not detect free disk; skipping check"
else
    if [ "$AVAIL_GB" -lt 20 ]; then
        fail "free disk in \$HOME is ${AVAIL_GB} GB; need >= 20 GB"
    else
        ok "free disk in \$HOME: ${AVAIL_GB} GB"
    fi
fi

# --- RAM ---
if [ -r /proc/meminfo ]; then
    TOTAL_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
    TOTAL_GB=$(( TOTAL_KB / 1024 / 1024 ))
    if [ "$TOTAL_GB" -lt 8 ]; then
        fail "total RAM is ${TOTAL_GB} GB; need >= 8 GB (recommended 16 GB)"
    elif [ "$TOTAL_GB" -lt 16 ]; then
        warn "total RAM is ${TOTAL_GB} GB; runs will be tight (recommend 16 GB)"
    else
        ok "total RAM: ${TOTAL_GB} GB"
    fi
else
    warn "/proc/meminfo not readable; skipping RAM check"
fi

# --- docker on PATH? ---
# Not a failure, but mention it. The pod is podman-only by policy.
if command -v docker >/dev/null 2>&1; then
    warn "docker found on PATH — this pod does NOT use docker. We use podman + Quadlet only."
fi

# --- systemd --user available? ---
if command -v systemctl >/dev/null 2>&1; then
    if systemctl --user list-units >/dev/null 2>&1; then
        ok "systemd --user reachable (Quadlet install supported)"
    else
        warn "systemd --user not reachable; Quadlet install will be skipped (runners still work)"
    fi
else
    warn "systemctl not found; Quadlet install will be skipped"
fi

if [ "$FAILED" -ne 0 ]; then
    printf '\n[preflight] one or more hard requirements failed. See [FAIL] lines above.\n'
    exit 1
fi

printf '\n[preflight] all checks passed.\n'
exit 0
