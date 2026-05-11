# Mazemaker Comparison Pod

A single-command Podman-based reproducer that runs Mazemaker plus five
competitor memory systems on the same benchmarks, then emits a
side-by-side comparison matrix. Hardfact-only: every number in the
verdict comes from a verifiable result JSON, or it is rendered as
`PENDING` / `ERROR` / `N/A`. No fabrication.

## Scope

**v0.1 (this release):** scaffolding + working Mazemaker reference
run. The dataset fetch, pod manifest, comparator, schema, verdict
template, and Mazemaker runner all work end-to-end. The five
competitor runners are stubs that import cleanly and respond to
`--help` but do not yet execute their target systems.

**v0.2 (next release):** Hindsight, Letta (formerly MemGPT), Mem0,
A-MEM, and Cognee runners. Each will follow the locked methodology
documented at `https://mazemaker.online/destruction/<system>/` and
emit `ResultRecord` JSON to `/work/results/<system>.json`.

## One-liner

```
bash <(curl -fsSL https://mazemaker.dev/bench.sh)
```

That entrypoint runs preflight checks (`scripts/detect_runtime.sh`),
fetches and hash-verifies the datasets, installs the Quadlet pod
manifest, and runs the comparator. Final output:

- `~/.bench-pod/results/*.json`
- `~/.bench-pod/matrix.md`
- `~/.bench-pod/matrix.json`
- `~/.bench-pod/verdict.md`

## From a clone

```
git clone https://github.com/itsXactlY/mazemaker
cd mazemaker/bench-pod
bash bench.sh                 # full matrix (Mazemaker + stubs)
bash bench.sh --only=mazemaker # Mazemaker reference run only
```

The script is idempotent and refuses to run on a missing pre-flight.
It never invokes `sudo`. It never invokes `docker`.

## Hardware floor

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| Disk (free) | 20 GB | 50 GB |
| CPU | x86_64, 4 cores | 8+ cores |
| GPU | optional | CUDA 12 (~6 GB VRAM) |

CUDA accelerates the embedding backend and the optional ColBERT rerank
channel. CPU-only is fully supported; latency rises but recall numbers
are identical.

## Runtime

This pod uses **Podman + Quadlet only**. There is no `docker`, no
`docker-compose`, no Kubernetes. Quadlet units land under
`~/.config/containers/systemd/bench/` and are managed via
`systemctl --user`.

LLM extraction for systems that need it (Mem0, A-MEM, Cognee) is
routed to an in-pod `ollama` instance. The `OPENAI_API_KEY` env var is
unset to ensure no traffic leaks to OpenAI.

## License

- **Mazemaker engine:** AGPLv3 (use, modify, study) AND PolyForm-NC-1.0
  (no commercial use without a Pro/Enterprise contract). See
  `LICENSE`, `LICENSE-AGPL-3.0.txt`, `LICENSE-POLYFORM-NC-1.0.0.md`
  in the parent repo.
- **A-MEM:** GPL — we do not vendor; the runner spawns the upstream
  project in its own container.
- **Hindsight, Letta, Mem0, Cognee:** their own licenses. The runners
  are thin shims; no upstream source is copied.
- **Benchmark datasets:** LongMemEval-S follows the upstream license
  at `xiaowu0162/long-mem-eval`. The synthetic_v1 dataset shipped in
  `datasets/bundled/` is original to this repo.

## Why use this

You have an agent. You want long-horizon memory. Six vendors will
sell you their box. None of them publish a reproducible matrix on a
neutral host. This pod is that matrix. Clone it, run it, see the
numbers on your hardware.

If the numbers convince you: the Pro pod runs the same engine on
Postgres + pgvector, with the Dream worker enabled and the Architect
console available at `https://mazemaker.online/onboard`. Free during
launch, self-host, single-tenant, no telemetry.
