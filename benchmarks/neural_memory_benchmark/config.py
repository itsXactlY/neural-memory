"""
Mazemaker Benchmark — Configuration
=======================================
All tunable knobs for the benchmark suite.
Default values are conservative; override via CLI or env vars.
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Resolve benchmark root relative to this file
BENCH_ROOT = Path(__file__).parent.resolve()
SRC_ROOT = BENCH_ROOT.parent.parent / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ── Mazemaker imports ────────────────────────────────────────────────────
# We import lazily inside functions to avoid breaking when the module isn't
# installed yet. Each suite handles its own import.


# ── Storage paths ─────────────────────────────────────────────────────────────
# Results live under the project's benchmarks/ dir (NOT /tmp and NOT the
# ~/.neural_memory* trees) so they're version-controllable and survive reboots.
# Override with NEURAL_BENCH_OUTPUT_DIR if you want a different destination.
_BENCH_DIR = BENCH_ROOT.parent  # benchmarks/
_DEFAULT_OUTPUT = Path(os.environ.get("NEURAL_BENCH_OUTPUT_DIR",
                                      str(_BENCH_DIR / "results")))


@dataclass
class Paths:
    """Where the benchmark stores its data."""
    output_dir: Path = field(default_factory=lambda: _DEFAULT_OUTPUT)
    results_dir: Path = field(default_factory=lambda: _DEFAULT_OUTPUT / "latest")
    data_dir: Path = field(default_factory=lambda: _DEFAULT_OUTPUT / "data")

    def ensure(self):
        for d in [self.output_dir, self.results_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Memory store ──────────────────────────────────────────────────────────────
# The benchmark MUST NEVER write to the production hot-path DB
# (~/.neural_memory/memory.db). The default below points at a benchmark sandbox
# directory; a guard in benchmark.py refuses to run if anything in the config
# resolves to the real hot path unless NEURAL_BENCH_ALLOW_HOTPATH=1 is set.
HOT_PATH_DB = (Path.home() / ".neural_memory" / "memory.db").resolve()
SANDBOX_DB = Path.home() / ".neural_memory_benchmark" / "sandbox.db"


@dataclass
class MemoryConfig:
    """Mazemaker under test."""
    db_path: str = os.environ.get(
        "NEURAL_BENCH_DB",
        str(SANDBOX_DB),
    )
    # Clear DB before benchmark run?  Default False = preserve existing data.
    clear_on_start: bool = False
    # How many connections to store per memory (max)
    max_connections: int = 50
    # Auto-commit interval (seconds)
    commit_interval: float = 5.0
    # Retrieval mode for default tests
    default_retrieval_mode: str = "semantic"
    # HNSW activation threshold
    hnsw_threshold: int = 5000


# ── Dataset generation ────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """Synthetic dataset generation parameters."""
    # Number of memories at each scale tier
    scale_tiers: list = field(default_factory=lambda: [100, 1_000, 10_000, 50_000, 100_000])
    # For "max" scale tier (push to limit)
    max_scale: int = 500_000

    # Per-generator memory counts
    episodic_count: int = 5_000
    factual_count: int = 3_000
    temporal_count: int = 2_000
    conversational_count: int = 1_000
    graph_node_count: int = 500
    adversarial_count: int = 500

    # Query generation
    queries_per_tier: int = 50
    # Ground truth: what fraction of memories should match a query?
    relevant_fraction: float = 0.05

    # Random seed for reproducibility
    seed: int = 42


# ── Retrieval benchmark ──────────────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    """Retrieval benchmark settings."""
    # Retrieval modes to test
    modes: list = field(default_factory=lambda: [
        "semantic", "hybrid", "advanced", "skynet"
    ])
    # Top-k values to test
    top_ks: list = field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    # Number of query repetitions for latency stability
    latency_runs: int = 5
    # Timeout per query (seconds)
    query_timeout: float = 30.0
    # Expected recall quality (answer must appear in top-k)
    recall_at_k: list = field(default_factory=lambda: [1, 3, 5])


# ── Dream Engine benchmark ───────────────────────────────────────────────────
@dataclass
class DreamConfig:
    """Dream consolidation benchmark."""
    # Phases to test
    phases: list = field(default_factory=lambda: ["nrem", "rem", "insight"])
    # Idle time before dreaming (seconds)
    idle_before_dream: float = 30.0
    # Minimum new memories to trigger dream
    dream_trigger_memories: int = 50
    # Number of dream cycles to measure
    dream_cycles: int = 3
    # Measure connection quality delta (before vs after)
    measure_quality_delta: bool = True


# ── GPU benchmark ────────────────────────────────────────────────────────────
@dataclass
class GPUConfig:
    """GPU recall performance settings."""
    batch_sizes: list = field(default_factory=lambda: [1, 8, 32, 64, 128, 256, 512])
    # Total embeddings to load for throughput test
    throughput_total: int = 100_000
    # Query count for latency test
    query_count: int = 10_000
    # Fall back to CPU if GPU unavailable?
    cpu_fallback: bool = False


# ── Scalability benchmark ─────────────────────────────────────────────────────
@dataclass
class ScalabilityConfig:
    """Scale-out benchmark settings."""
    tiers: list = field(default_factory=lambda: [1_000, 10_000, 50_000, 100_000, 500_000])
    # Step size for incremental growth
    step_size: int = 10_000
    # Measure retrieval time vs memory count
    measure_retrieval_vs_count: bool = True
    # Measure WAL file growth
    measure_wal_growth: bool = True


# ── Graph benchmark ──────────────────────────────────────────────────────────
@dataclass
class GraphConfig:
    """Graph traversal benchmark."""
    # Traversal engines to compare
    engines: list = field(default_factory=lambda: ["bfs", "ppr"])
    # Graph depths to test
    depths: list = field(default_factory=lambda: [2, 3, 5, 8])
    # Number of starting nodes per depth
    start_nodes_per_depth: int = 20
    # PPR teleport probability
    ppr_teleport: float = 0.15
    # PPR damping factor
    ppr_damping: float = 0.85
    # PPR iterations
    ppr_iterations: int = 50


# ── Concurrent benchmark ─────────────────────────────────────────────────────
@dataclass
class ConcurrentConfig:
    """Multi-threaded stress test."""
    # Number of concurrent writers
    writer_threads: list = field(default_factory=lambda: [1, 2, 4, 8, 16])
    # Number of concurrent readers
    reader_threads: list = field(default_factory=lambda: [1, 4, 8, 16, 32])
    # Operations per thread
    ops_per_thread: int = 100
    # WAL size check interval
    wal_check_interval: float = 1.0
    # Max acceptable WAL size (MB)
    max_acceptable_wal_mb: float = 100.0


# ── Conflict benchmark ───────────────────────────────────────────────────────
@dataclass
class ConflictConfig:
    """Conflict detection and supersession."""
    # Number of conflicting memory groups
    conflict_groups: int = 20
    # Memories per conflict group
    memories_per_group: int = 5
    # Measure recall degradation after conflict resolution
    measure_recall_degradation: bool = True


# ── Agentic workflow benchmark ────────────────────────────────────────────────
@dataclass
class AgenticConfig:
    """End-to-end agent workflow simulation."""
    # Number of simulated sessions
    sessions: int = 5
    # Average turns per session
    turns_per_session: int = 20
    # Actions per turn (remember, recall, think, graph)
    actions_per_turn: int = 4
    # Measure end-to-end latency
    measure_e2e_latency: bool = True


# ── Global config ────────────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    """Top-level config that composes all sub-configs."""
    paths: Paths = field(default_factory=Paths)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dream: DreamConfig = field(default_factory=DreamConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    scalability: ScalabilityConfig = field(default_factory=ScalabilityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    concurrent: ConcurrentConfig = field(default_factory=ConcurrentConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)

    # Suites to run (empty = all)
    suites: list = field(default_factory=list)

    # Dry-run: generate data but don't execute
    dry_run: bool = False

    # Use the disjoint-vocabulary paraphrase dataset (dataset_v2) for queries
    # instead of the bag-of-tokens query generator. Required for the v2
    # suites that depend on 1:1 ground truth.
    paraphrase: bool = False

    # Use the real-text dataset (dataset_real) — chunks pulled from the
    # project's own .md/.py prose, anchored by real CamelCase / snake_case
    # / *.py tokens. Addresses codex v5's "synthetic data only" caveat.
    # Wins over `paraphrase` if both are set.
    realistic: bool = False

    # Report format: "text" | "json" | "both"
    report_format: str = "both"

    # LLM model to use for generation (None = no LLM, synthetic only)
    llm_model: str = os.environ.get("OPENAI_API_KEY", None) and "gpt-4o"

    @classmethod
    def from_args(cls, args=None):
        """Build config from CLI args or defaults."""
        cfg = cls()
        cfg.paths.ensure()
        return cfg
