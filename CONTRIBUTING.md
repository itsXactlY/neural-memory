# Contributing to Mazemaker Adapter

## Quick Start for Contributors

```bash
# Clone
git clone https://github.com/itsXactlY/mazemaker-adapter.git
cd mazemaker-adapter

# Install dev dependencies
pip install sentence-transformers numpy pytest

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run tests
cd ../python
python3 test_suite.py
```

## Project Structure

```
mazemaker-adapter/
├── include/neural/    # C++ headers (public API)
├── src/               # C++ implementation
├── python/            # Python bindings + client
├── sql/               # Database schemas
├── tests/             # C++ unit tests
└── .github/           # CI + issue templates
```

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/itsXactlY/mazemaker-adapter/issues)
2. Create a new issue with:
   - OS + Python version
   - Steps to reproduce
   - Expected vs actual behavior
   - Stack trace if applicable

### Suggesting Features

Open an issue with the `enhancement` label. Include:
- Use case (why you need this)
- Proposed API (how it would work)
- Any relevant research papers

### Code Contributions

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests first (TDD encouraged)
4. Implement the feature
5. Run tests: `python3 python/test_suite.py`
6. Run benchmarks: `cd build && ./bench_vector_ops`
7. Submit a PR

### Code Style

**C++:**
- C++23 standard
- `snake_case` for functions/variables
- `PascalCase` for classes
- 4-space indent
- Braces on same line
- Always use `{}` for if/else

**Python:**
- PEP 8
- Type hints on public APIs
- Docstrings on public classes/methods
- No trailing whitespace

### Testing

```bash
# All tests
python3 python/test_suite.py

# By tag
python3 python/test_suite.py --tags embed
python3 python/test_suite.py --tags memory,threading
python3 python/test_suite.py --tags perf

# List available tags
python3 python/test_suite.py --list-tags

# C++ tests
cd build && ./test_vector_ops && ./test_hopfield && ./test_graph

# Benchmarks
cd build && ./bench_vector_ops
```

### Test Tags

| Tag | Description |
|-----|-------------|
| `embed` | Embedding provider tests |
| `storage` | SQLite/Postgres storage tests |
| `memory` | Memory client tests |
| `graph` | Knowledge graph tests |
| `threading` | Thread safety tests |
| `api` | Unified API tests |
| `cpp` | C++ library tests |
| `hermes` | Hermes plugin tests |
| `perf` | Performance benchmarks |
| `stress` | Stress/load tests |
| `slow` | Tests that take > 1s |

## Architecture Decisions

### Why Hopfield Networks?

Modern Hopfield Networks are mathematically equivalent to Transformer attention ([Ramsauer et al., 2020](https://arxiv.org/abs/2008.02217)). This means:
- Proven math (attention is well-understood)
- No exotic hardware (runs on CPU)
- Exponential storage capacity
- Natural connection discovery via self-attention

### Why SQLite + Postgres?

- **SQLite**: Fast local storage, zero config, works everywhere
- **Postgres + pgvector**: Pro-tier deployments, shared memory across agents, ACID, native ARM

### Why Python + C++?

- **Python**: Rapid prototyping, ML ecosystem (sentence-transformers)
- **C++**: SIMD-accelerated vector ops, sub-millisecond latency

## Adding a New Feature

### New Embedding Backend

1. Add class in `python/embed_provider.py`:
   ```python
   class MyBackend:
       def __init__(self, dim=1024): ...
       def embed(self, text: str) -> list[float]: ...
       def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
   ```

2. Register in `auto_detect()`:
   ```python
   try:
       return MyBackend()
   except ImportError:
       pass
   ```

3. Add tests in `python/test_suite.py`

### New Storage Backend

1. Implement `SQLiteStore` interface in `python/`
2. Update `neural_memory.py` to use it
3. Add tests

### New C++ Function

1. Add declaration in `include/neural/c_api.h`
2. Add implementation in `src/core/c_api.cpp`
3. Update `python/cpp_bridge.py` ctypes wrapper
4. Add tests

## Performance Guidelines

- **Vector ops**: Must be SIMD-accelerated (AVX2 minimum)
- **Embedding**: Must handle batch operations
- **Storage**: Must be thread-safe
- **Memory**: Must not leak (use RAII in C++)
- **Latency**: < 1ms for similarity search on 100 vectors

## Release Process

1. Update version in `CMakeLists.txt` and `setup.py`
2. Run full test suite
3. Run benchmarks
4. Update CHANGELOG.md
5. Tag: `git tag v1.0.0`
6. Push: `git push --tags`
7. Create GitHub release

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/itsXactlY/mazemaker-adapter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/itsXactlY/mazemaker-adapter/discussions)
- **Discord**: #mazemaker in Hermes Discord

## License

MIT - see [LICENSE](LICENSE)
