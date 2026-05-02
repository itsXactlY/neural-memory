"""Tests for Spanish→English query translator + opt-in env var wiring.

Per 2026-05-02 ship: dict-based translator gives +0.045 R@5 on AE-domain
bench by translating Spanish queries to English so they hit English-content
substrate. Opt-in via NM_SPANISH_TRANSLATE=1.

Locks in the contract so future "improvements" don't silently regress
the cable-doce hit (which I demonstrated could happen — see commit
message of f1c0b64).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class SpanishTranslatorTests(unittest.TestCase):
    """Test the _translate_spanish_to_english classmethod directly."""

    def test_empty_query_returns_empty(self) -> None:
        self.assertEqual(NeuralMemory._translate_spanish_to_english(""), "")

    def test_pure_english_passes_through(self) -> None:
        # Unknown words pass through; "find panels" would be untouched
        # because none of those tokens are in the Spanish dict
        result = NeuralMemory._translate_spanish_to_english("find panels at lot 12")
        # Lowercases the words; "find/panels/at/lot/12" not in dict → unchanged
        # except case
        self.assertIn("panels", result.lower())
        self.assertIn("12", result)

    def test_cable_doce_translation_preserves_search_signal(self) -> None:
        """The query that gives the +0.045 R@5 lift. Don't break this."""
        result = NeuralMemory._translate_spanish_to_english(
            "Busca mensaje de WhatsApp sobre cable numero doce."
        )
        # Must contain key search tokens that hit English WA bridge memories
        self.assertIn("search", result.lower())
        self.assertIn("message", result.lower())
        self.assertIn("whatsapp", result.lower())
        self.assertIn("wire", result.lower())  # cable → wire
        self.assertIn("number", result.lower())  # numero → number
        self.assertIn("twelve", result.lower())  # doce → twelve

    def test_material_que_no_llego_translates_known_tokens(self) -> None:
        """Spanish bench query about missing material."""
        result = NeuralMemory._translate_spanish_to_english(
            "Encuentra la conversacion sobre material que no llego."
        )
        self.assertIn("find", result.lower())  # encuentra → find
        self.assertIn("about", result.lower())  # sobre → about
        self.assertIn("material", result.lower())  # passes through
        self.assertIn("not", result.lower())  # no → not
        self.assertIn("arrived", result.lower())  # llego → arrived

    def test_unknown_tokens_pass_through_unchanged(self) -> None:
        """SKUs, names, numbers must not get mangled."""
        result = NeuralMemory._translate_spanish_to_english("HD299898 cable doce")
        self.assertIn("HD299898", result)  # SKU preserved (case insensitive lookup but value preserved)

    def test_punctuation_preserved(self) -> None:
        """Trailing punctuation should survive."""
        result = NeuralMemory._translate_spanish_to_english("Que se dijo?")
        self.assertTrue(result.endswith("?"))

    def test_whatsapp_loanword_unchanged(self) -> None:
        """Common loanwords (WhatsApp, breakers) shouldn't be re-translated."""
        result = NeuralMemory._translate_spanish_to_english("comprar breakers WhatsApp")
        self.assertIn("buy", result.lower())  # comprar → buy
        self.assertIn("breakers", result.lower())
        self.assertIn("whatsapp", result.lower())


class SpanishDetectionTests(unittest.TestCase):
    """The translator only fires when Spanish is detected via
    _should_skip_rerank. Verify detection still works."""

    def test_pure_english_not_flagged(self) -> None:
        self.assertFalse(NeuralMemory._should_skip_rerank("find panels at lot 12"))

    def test_spanish_with_diacritics_flagged(self) -> None:
        # Single non-ASCII char triggers (layer 1)
        self.assertTrue(NeuralMemory._should_skip_rerank("¿Qué pasó?"))

    def test_spanish_without_diacritics_flagged(self) -> None:
        # Layer 2: 2+ Spanish indicator words
        self.assertTrue(NeuralMemory._should_skip_rerank(
            "Encuentra la conversacion sobre material que no llego"
        ))

    def test_english_with_one_spanish_word_not_flagged(self) -> None:
        # "no" alone shouldn't flag (low false-positive)
        self.assertFalse(NeuralMemory._should_skip_rerank("no panel found"))


class SpanishDictRegressionTests(unittest.TestCase):
    """Lock in dict entries we know are working. If anyone removes one,
    the bench-validated lift could disappear silently."""

    def test_critical_translations_present(self) -> None:
        """These specific mappings underlie the cable-doce R@5 hit."""
        d = NeuralMemory._SPANISH_TO_ENGLISH
        self.assertEqual(d["cable"], "wire")  # essential for WA crew msgs
        self.assertEqual(d["numero"], "number")
        self.assertEqual(d["doce"], "twelve")
        self.assertEqual(d["busca"], "search")
        self.assertEqual(d["mensaje"], "message")
        self.assertEqual(d["sobre"], "about")
        self.assertEqual(d["encuentra"], "find")
        self.assertEqual(d["material"], "material")  # passthrough kept

    def test_de_not_remapped_to_of(self) -> None:
        """Lesson learned 2026-05-02 (commit f1c0b64): mapping de→of broke
        the cable-doce hit. Don't re-introduce."""
        d = NeuralMemory._SPANISH_TO_ENGLISH
        # Either not present, or maps to itself / something other than 'of'
        if "de" in d:
            self.assertNotEqual(
                d["de"], "of",
                "de→of regressed R@5 from 0.6818 to 0.6364 on cable-doce. "
                "Keep removed or use a different mapping.",
            )


if __name__ == "__main__":
    unittest.main()
