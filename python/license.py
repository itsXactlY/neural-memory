"""
license.py — Mazemaker Pro license verification.

The engine ships under a dual AGPLv3 + PolyForm-NC license.  The
*community* build (anyone who clones the repo) provides hybrid recall,
NREM-only consolidation, SQLite WAL, and the CLI / MCP server.  Paid
tiers (Pro, Enterprise) unlock additional features by carrying a signed
license JWT issued by the operator's license-validation backend.

Honesty rules — these are non-negotiable:

  1. Every gate in the engine is a plain ``if has_feature(name)`` call.
     Grep-friendly, reviewable, no obfuscation.
  2. The community build is real: it must build, run, pass its tests,
     and reproduce the published R@5 = 0.96 hybrid number on
     LongMemEval-S without any license at all.
  3. Verification failures degrade gracefully back to community — never
     silently into Pro.  A failure logs once, loud and clear.
  4. The license check runs once at startup and caches the result.
     Zero per-call overhead in recall / ingest hot paths.
  5. AGPL §6 compliance: the source is buildable and the engine works.
     The license-key gate is a feature switch, not an installation gate.

JWT format
----------

The engine accepts the same JWT shape minted by the operator's
license-validation backend (see backend/client/pod/mazemaker/shared/
license_schema.py for the canonical Pydantic model).  Required claims:

* iss        = "mazemaker.dev"
* aud        = "mazemaker-pod"
* tier       ∈ {free, payg, pro, enterprise}
* exp, nbf, iat, jti, sub, kid, user_id
* grace_until = exp + 7 days
* backend    ∈ {sqlite, postgres}

Tier → feature mapping (engine-side derivation):

* community (no JWT)         → no Pro features
* free                       → no Pro features
* payg (Lite)                → no Pro features (Lite = managed install
                                of the community feature set)
* pro                        → all features
* enterprise                 → all features

The Postgres backend is additionally gated by ``claims.backend ==
"postgres"`` — Pro/Enterprise tiers can opt-in or out via JWT.

Loading order
-------------

Env overrides take precedence over the default file path so tests +
operator tooling can inject a token without disturbing the on-disk one.

1. ``MM_LICENSE_JWT`` env (raw token).
2. ``MM_LICENSE_JWT_PATH`` env (explicit path).
3. ``MAZEMAKER_LICENSE_PATH`` env (matches the v2 pod convention).
4. ``~/.mazemaker/license.jwt`` (default; what install.sh writes).
5. None of the above → community-tier feature set.

Pubkey loading order
--------------------

1. ``MM_LICENSE_PUBKEY_PATH`` env (PEM file path, useful for
   Enterprise air-gap deployments running their own license server).
2. ``MAZEMAKER_PUBKEY_PATH`` env (matches the v2 pod convention).
3. ``OPERATOR_PUBKEY_HEX`` embedded in this module (default — the
   mazemaker.dev v1 license-signing public key).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("mazemaker.license")

# ---------------------------------------------------------------------------
# The operator's Ed25519 license-signing public key (raw, 32-byte hex).
#
# This is a *public* key — embedding it here is intentional.  Every
# clone of the engine must be able to verify a license, but nobody can
# forge one without the matching private key (held only by the
# operator's license-validation backend at api.mazemaker.dev).
#
# kid: "v1"
# corresponding PEM lives at /secrets/jwt.v1.pub.ed25519 in the
# operator-managed pod images.
# ---------------------------------------------------------------------------
OPERATOR_PUBKEY_HEX = (
    "15ab2e9bdfbfba31a39b1f50668c6921231d6990b52b0ef3bebe4046bbf63f3c"
)

JWT_AUDIENCE = "mazemaker-pod"
JWT_ALGORITHM = "EdDSA"
JWT_ISSUER = "mazemaker.dev"

# Tiers that grant Pro features.  "free" and "payg" are paid/onboarded
# but get the same engine feature set as community — the Lite (payg)
# tier upsells managed install + email support, not engine features.
PRO_TIERS = frozenset({"pro", "enterprise"})

# All feature names recognised by the engine.  Additions are
# forward-compatible (new gate sites can be introduced without
# breaking older license-issuing servers).
KNOWN_FEATURES = frozenset({
    "colbert",       # ColBERT@1.5 late-interaction rerank channel
    "rem",           # REM dream phase (bridge-memory creation)
    "insight",       # Insight dream phase (cluster summary memories)
    "architect",     # Architect UI server-side data API
    "dream_worker",  # Autonomous overnight dream-worker loop
    "postgres",      # Postgres + pgvector backend adapter
})


# ---------------------------------------------------------------------------
# License object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class License:
    """Parsed license state.  Use :func:`has_feature` for gate checks."""

    tier: str = "community"
    features: frozenset = field(default_factory=frozenset)
    sub: Optional[str] = None       # device fingerprint
    user_id: Optional[str] = None
    iat: int = 0
    exp: int = 0
    grace_until: int = 0
    backend: str = "sqlite"          # sqlite | postgres

    @property
    def is_community(self) -> bool:
        return self.tier == "community"

    @property
    def is_in_grace(self) -> bool:
        if self.exp == 0 or self.grace_until == 0:
            return False
        now = time.time()
        return self.exp < now <= self.grace_until

    @property
    def is_past_grace(self) -> bool:
        if self.grace_until == 0:
            return False
        return time.time() > self.grace_until

    def has(self, feature: str) -> bool:
        if self.is_past_grace:
            return False
        return feature in self.features

    def __str__(self) -> str:
        if self.is_community:
            return "License(community)"
        feats = ",".join(sorted(self.features)) or "none"
        return (f"License(tier={self.tier}, features=[{feats}], "
                f"backend={self.backend}, exp={self.exp})")


# ---------------------------------------------------------------------------
# Tier → feature derivation
# ---------------------------------------------------------------------------

def _features_for_tier(tier: str, backend: str) -> frozenset:
    """Map a tier (and storage backend) to the engine feature set.

    Pro / Enterprise tiers receive every Pro engine feature.  The
    Postgres gate is additionally controlled by the JWT's ``backend``
    field — a Pro user with ``backend=sqlite`` runs ColBERT + REM +
    Insight + Architect + dream-worker but stays on SQLite.
    """
    if tier not in PRO_TIERS:
        return frozenset()

    feats = {"colbert", "rem", "insight", "architect", "dream_worker"}
    if backend == "postgres":
        feats.add("postgres")
    return frozenset(feats)


# ---------------------------------------------------------------------------
# Token + pubkey loading
# ---------------------------------------------------------------------------

def _read_license_token() -> Optional[str]:
    """Resolve the license JWT according to the documented loading order."""
    raw = os.getenv("MM_LICENSE_JWT")
    if raw and raw.strip():
        return raw.strip()

    for env_var in ("MM_LICENSE_JWT_PATH", "MAZEMAKER_LICENSE_PATH"):
        explicit = os.getenv(env_var)
        if explicit:
            p = Path(explicit).expanduser()
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
            log.warning("%s=%s does not exist", env_var, explicit)

    default_path = Path("~/.mazemaker/license.jwt").expanduser()
    if default_path.is_file():
        return default_path.read_text(encoding="utf-8").strip()

    return None


def _load_pubkey_pem() -> Optional[bytes]:
    """Resolve the operator pubkey as a PEM-encoded byte string.

    Returns None if the embedded hex is unparseable AND no env override
    is provided (extremely unlikely; would indicate a tampered build).
    """
    for env_var in ("MM_LICENSE_PUBKEY_PATH", "MAZEMAKER_PUBKEY_PATH"):
        path = os.getenv(env_var)
        if path:
            p = Path(path).expanduser()
            if p.is_file():
                return p.read_bytes()
            log.warning("%s=%s does not exist", env_var, path)

    # Convert the embedded raw-hex pubkey to PEM so PyJWT (which expects
    # a key object) accepts it.
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        pub = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(OPERATOR_PUBKEY_HEX)
        )
        return pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except ImportError:
        log.warning(
            "cryptography not installed — license verification is "
            "unavailable; engine will run in community mode. "
            "Install with: pip install 'cryptography>=41.0.0'"
        )
        return None
    except Exception as e:
        log.warning("embedded pubkey conversion failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# The actual verifier
# ---------------------------------------------------------------------------

def _verify_token(token: str, pubkey_pem: bytes) -> Optional[dict]:
    """Verify and decode the JWT.  Returns decoded claims or None."""
    try:
        import jwt as pyjwt
    except ImportError:
        log.warning(
            "PyJWT not installed — license verification unavailable; "
            "engine will run in community mode. "
            "Install with: pip install 'PyJWT>=2.8.0'"
        )
        return None

    try:
        # Read grace_until without verification to set leeway.
        unverified = pyjwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False,
                     "verify_aud": False},
            algorithms=[JWT_ALGORITHM],
        )
        exp = int(unverified.get("exp", 0))
        grace_until = int(unverified.get("grace_until", 0))
        leeway = float(max(0, grace_until - exp))
    except Exception as e:
        log.warning("license decode (peek) failed: %s", e)
        return None

    try:
        return pyjwt.decode(
            token,
            pubkey_pem,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            options={"verify_exp": True, "verify_aud": True},
            leeway=leeway,
        )
    except pyjwt.ExpiredSignatureError:
        log.warning("license is past grace period (exp + 7 days)")
        return None
    except pyjwt.InvalidSignatureError:
        log.warning("license signature is invalid")
        return None
    except Exception as e:
        log.warning("license verification failed: %s", e)
        return None


def load_license(pubkey_pem: Optional[bytes] = None) -> License:
    """Load + verify the operator-issued license.  Returns a community
    :class:`License` on any failure, never raises.

    Tests may pass a different ``pubkey_pem`` to verify against an
    in-test keypair.  Normal callers should use :func:`has_feature`.
    """
    token = _read_license_token()
    if token is None:
        return License()  # community

    if pubkey_pem is None:
        pubkey_pem = _load_pubkey_pem()
    if pubkey_pem is None:
        return License()

    claims = _verify_token(token, pubkey_pem)
    if claims is None:
        log.warning(
            "license verification failed; engine is running in community "
            "mode (hybrid recall, NREM-only dream, SQLite, CLI/MCP). "
            "Pro features (ColBERT, REM/Insight phases, Architect UI, "
            "dream-worker, Postgres) are disabled."
        )
        return License()

    tier = str(claims.get("tier", "community"))
    backend = str(claims.get("backend", "sqlite"))
    features = _features_for_tier(tier, backend)

    license_ = License(
        tier=tier,
        features=features,
        sub=claims.get("sub"),
        user_id=claims.get("user_id"),
        iat=int(claims.get("iat", 0)),
        exp=int(claims.get("exp", 0)),
        grace_until=int(claims.get("grace_until", 0)),
        backend=backend,
    )

    log.info("loaded %s", license_)
    return license_


# ---------------------------------------------------------------------------
# Module-singleton API used by gate sites throughout the engine
# ---------------------------------------------------------------------------

_loaded: Optional[License] = None


def get_license() -> License:
    """Return the singleton :class:`License`, loading it on first call."""
    global _loaded
    if _loaded is None:
        _loaded = load_license()
    return _loaded


def has_feature(name: str) -> bool:
    """Cheap, hot-path-safe feature gate.

    Use this at the call sites that gate Pro features::

        from license import has_feature

        if has_feature("colbert"):
            score += colbert_score * weight

    No exception path — returns False for any unknown / malformed
    state.  Logs are emitted only at :func:`load_license` time, never
    in this function.
    """
    return get_license().has(name)


def reset_for_tests() -> None:
    """Test-only: drop the cached singleton so the next call reloads."""
    global _loaded
    _loaded = None
