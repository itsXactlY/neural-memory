#!/usr/bin/env python3
"""
Standalone Embedding Server for Neural Memory.

Run as a persistent background process — survives hermes-agent restarts.
Shares the model across all processes via UNIX socket.

Usage:
    python3 embed-server.py                    # foreground (for testing)
    nohup python3 embed-server.py &            # background
    systemctl --user start neural-embed       # via systemd

Env vars:
  EMBED_MODEL        — model name (default: BAAI/bge-m3)
  EMBED_DEVICE       — device: cuda/cpu/mps (default: auto)
  EMBED_IDLE_TIMEOUT — seconds before GPU->CPU eject (0=disabled, default: 300)
  EMBED_SOCKET       — UNIX socket path
"""
import os
import sys
import signal
import argparse
import time

# Add python/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embed_provider import SharedEmbedServer, SOCKET_PATH


def main():
    parser = argparse.ArgumentParser(description='Neural Memory Embedding Server')
    parser.add_argument('--model', default=os.environ.get('EMBED_MODEL', 'BAAI/bge-m3'),
                        help='Model name (default: BAAI/bge-m3)')
    parser.add_argument('--device', default=os.environ.get('EMBED_DEVICE', None),
                        help='Device: cuda/cpu/mps (default: auto)')
    parser.add_argument('--idle-timeout', type=int, default=300,
                        help='Seconds before GPU->CPU eject (0=disabled, default: 300)')
    args = parser.parse_args()

    print("=" * 60)
    print("Neural Memory Embed Server")
    print("  Model:        {0}".format(args.model))
    print("  Device:       {0}".format(args.device or 'auto'))
    print("  Idle timeout: {0}s (0=disabled)".format(args.idle_timeout))
    print("  Socket:       {0}".format(SOCKET_PATH))
    print("  PID:          {0}".format(os.getpid()))
    print("=" * 60)

    # Clean up stale socket
    if SOCKET_PATH.exists():
        print("[embed-server] Removing stale socket: {0}".format(SOCKET_PATH))
        SOCKET_PATH.unlink()

    server = SharedEmbedServer(
        model_name=args.model,
        device=args.device,
        idle_timeout=args.idle_timeout,
    )

    def shutdown(signum, frame):
        print("\n[embed-server] Shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if not server.start():
        print("[embed-server] Server already running or failed to start.")
        sys.exit(1)

    print("[embed-server] Ready. Press Ctrl+C to stop.")

    # Block forever — server runs in background threads
    while True:
        time.sleep(86400)


if __name__ == '__main__':
    main()
