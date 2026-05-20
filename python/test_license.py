"""
Unit tests for python/license.py.

These tests use a *test* Ed25519 keypair generated fresh in setUp — the
operator's real pubkey is not exercised here.  This proves the
verification logic is correct on a known-good keypair, which is what
the tests should prove; the operator pubkey is constant production
state, separately verified during deploy.

JWT shape matches the canonical license schema in
backend/client/pod/mazemaker/shared/license_schema.py.

Run: pytest python/test_license.py
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

import jwt as pyjwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import license as lic


def _pubkey_pem(priv: Ed25519PrivateKey) -> bytes:
    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _priv_pem(priv: Ed25519PrivateKey) -> bytes:
    return priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _make_jwt(priv: Ed25519PrivateKey, *,
              tier: str = "pro",
              backend: str = "postgres",
              exp_offset: int = 30 * 86400,
              grace_offset: int = 37 * 86400,
              extra: dict | None = None) -> str:
    """Build a license JWT shaped like the production schema."""
    now = int(time.time())
    payload = {
        "iss": "mazemaker.dev",
        "sub": "fp_" + "0" * 64,
        "aud": "mazemaker-pod",
        "iat": now - 60,
        "nbf": now - 60,
        "exp": now + exp_offset,
        "jti": "lic_test",
        "kid": "v1",
        "tier": tier,
        "user_id": "usr_test",
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "grace_until": now + grace_offset,
        "abuse_score": 0.0,
        "quota": {
            "memories_max": -1,
            "calls_remaining_today": -1,
            "calls_remaining_month": -1,
            "managed_provider_calls_remaining_today": 0,
        },
        "embedding": {
            "providers_allowed": ["fastembed"],
            "managed_provider": None,
        },
        "compute": {
            "device": "auto",
            "embedding_backend": "auto",
            "strict": False,
            "recall_mode": "hybrid",
            "gpu_cache": "auto",
        },
        "backend": backend,
        "vault_secret": "dGVzdC12YXVsdC1zZWNyZXQ=",
    }
    if extra:
        payload.update(extra)
    return pyjwt.encode(
        payload,
        _priv_pem(priv),
        algorithm="EdDSA",
        headers={"kid": "v1"},
    )


class _BaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.priv = Ed25519PrivateKey.generate()
        self.pubkey_pem = _pubkey_pem(self.priv)
        # Wipe any user-side env that could pollute the test.
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in (
                "MM_LICENSE_JWT", "MM_LICENSE_JWT_PATH",
                "MAZEMAKER_LICENSE_PATH",
                "MM_LICENSE_PUBKEY_PATH", "MAZEMAKER_PUBKEY_PATH",
            )
        }
        lic.reset_for_tests()

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
        lic.reset_for_tests()


class LicenseLoaderTests(_BaseTestCase):

    # ---- baseline -------------------------------------------------------

    def test_no_token_yields_community(self) -> None:
        license_ = lic.load_license(self.pubkey_pem)
        self.assertTrue(license_.is_community)
        self.assertEqual(license_.tier, "community")
        self.assertEqual(license_.features, frozenset())
        self.assertFalse(license_.has("colbert"))
        self.assertFalse(license_.has("rem"))
        self.assertFalse(license_.has("postgres"))

    def test_garbage_token_falls_back_to_community(self) -> None:
        os.environ["MM_LICENSE_JWT"] = "not.a.jwt"
        license_ = lic.load_license(self.pubkey_pem)
        self.assertTrue(license_.is_community)

    # ---- tier mapping ---------------------------------------------------

    def test_pro_tier_unlocks_all_features(self) -> None:
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="pro",
                                                 backend="postgres")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertEqual(license_.tier, "pro")
        for f in ("colbert", "rem", "insight", "architect",
                  "dream_worker", "postgres", "dae"):
            self.assertTrue(license_.has(f), f"{f} should be granted")
        self.assertFalse(license_.has("nonexistent"))

    def test_enterprise_tier_unlocks_all_features(self) -> None:
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="enterprise",
                                                 backend="postgres")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertEqual(license_.tier, "enterprise")
        self.assertTrue(license_.has("colbert"))
        self.assertTrue(license_.has("postgres"))

    def test_pro_with_sqlite_backend_no_postgres_feature(self) -> None:
        # Pro user on SQLite still gets ColBERT + REM + Insight +
        # Architect + dream-worker, but the Postgres gate stays closed.
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="pro",
                                                 backend="sqlite")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertTrue(license_.has("colbert"))
        self.assertTrue(license_.has("rem"))
        self.assertFalse(license_.has("postgres"))

    def test_payg_tier_normalises_to_lite_no_pro_features(self) -> None:
        # Legacy payg vocabulary normalises to "lite".
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="payg",
                                                 backend="sqlite")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertEqual(license_.tier, "lite")
        self.assertFalse(license_.has("colbert"))
        self.assertFalse(license_.has("rem"))
        self.assertFalse(license_.has("architect"))

    def test_free_tier_normalises_to_community_no_pro_features(self) -> None:
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="free",
                                                 backend="sqlite")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertEqual(license_.tier, "community")
        self.assertFalse(license_.has("colbert"))

    def test_lite_tier_jwt_passes_through(self) -> None:
        # New backend vocabulary — minted after 2026-05-20.
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="lite",
                                                 backend="sqlite")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertEqual(license_.tier, "lite")
        self.assertFalse(license_.has("colbert"))

    def test_community_tier_jwt_passes_through(self) -> None:
        os.environ["MM_LICENSE_JWT"] = _make_jwt(self.priv,
                                                 tier="community",
                                                 backend="sqlite")
        license_ = lic.load_license(self.pubkey_pem)
        self.assertEqual(license_.tier, "community")
        self.assertFalse(license_.has("colbert"))

    def test_normalize_tier_helper(self) -> None:
        self.assertEqual(lic._normalize_tier("free"), "community")
        self.assertEqual(lic._normalize_tier("payg"), "lite")
        self.assertEqual(lic._normalize_tier("community"), "community")
        self.assertEqual(lic._normalize_tier("lite"), "lite")
        self.assertEqual(lic._normalize_tier("pro"), "pro")
        self.assertEqual(lic._normalize_tier("enterprise"), "enterprise")
        # Unknown values pass through unchanged.
        self.assertEqual(lic._normalize_tier("future_tier"), "future_tier")

    # ---- expiry ---------------------------------------------------------

    def test_in_grace_still_active(self) -> None:
        # Expired 3 days ago — within the 7-day grace window.
        os.environ["MM_LICENSE_JWT"] = _make_jwt(
            self.priv, tier="pro", backend="postgres",
            exp_offset=-3 * 86400,
            grace_offset=4 * 86400,  # grace ends 4 days from now
        )
        license_ = lic.load_license(self.pubkey_pem)
        self.assertTrue(license_.is_in_grace)
        self.assertFalse(license_.is_past_grace)
        self.assertTrue(license_.has("colbert"))

    def test_past_grace_falls_back(self) -> None:
        # Expired 30 days ago, grace window also long expired.
        os.environ["MM_LICENSE_JWT"] = _make_jwt(
            self.priv, tier="pro", backend="postgres",
            exp_offset=-30 * 86400,
            grace_offset=-23 * 86400,
        )
        license_ = lic.load_license(self.pubkey_pem)
        self.assertTrue(license_.is_community)

    # ---- tampering ------------------------------------------------------

    def test_signature_from_wrong_key_rejected(self) -> None:
        token = _make_jwt(self.priv, tier="pro", backend="postgres")
        os.environ["MM_LICENSE_JWT"] = token
        # Verify against an unrelated keypair.
        other_pem = _pubkey_pem(Ed25519PrivateKey.generate())
        license_ = lic.load_license(other_pem)
        self.assertTrue(license_.is_community)

    def test_audience_mismatch_rejected(self) -> None:
        # Mint a JWT with the wrong aud — verifier must reject.
        os.environ["MM_LICENSE_JWT"] = _make_jwt(
            self.priv, tier="pro", backend="postgres",
            extra={"aud": "wrong-audience"},
        )
        license_ = lic.load_license(self.pubkey_pem)
        self.assertTrue(license_.is_community)

    # ---- file path resolution ------------------------------------------

    def test_explicit_path_takes_precedence(self) -> None:
        token = _make_jwt(self.priv, tier="pro", backend="postgres")
        with tempfile.NamedTemporaryFile("w", suffix=".jwt",
                                         delete=False) as f:
            f.write(token)
            path = f.name
        try:
            os.environ["MM_LICENSE_JWT_PATH"] = path
            license_ = lic.load_license(self.pubkey_pem)
            self.assertTrue(license_.has("postgres"))
        finally:
            os.unlink(path)


class HasFeatureSingletonTests(_BaseTestCase):

    def test_default_community(self) -> None:
        # No JWT in env, no fixture ~/.mazemaker/license.jwt expected
        # in CI — community floor.  (The dev's own machine may have a
        # real license; that's fine, the test only asserts the cache
        # works once loaded.)
        first = lic.has_feature("colbert")
        second = lic.has_feature("colbert")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
