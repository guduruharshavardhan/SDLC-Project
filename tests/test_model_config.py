import os
import unittest
from unittest.mock import patch

from agents.crew_pipeline import DEFAULT_GROQ_MODEL, _resolve_groq_model


class ResolveGroqModelTests(unittest.TestCase):
    def test_uses_replacement_model_by_default(self) -> None:
        with patch.dict(os.environ, {"GROQ_MODEL": ""}):
            self.assertEqual(_resolve_groq_model(), DEFAULT_GROQ_MODEL)

    def test_replaces_deprecated_configured_model(self) -> None:
        with patch.dict(os.environ, {"GROQ_MODEL": "llama-3.1-8b-instant"}):
            self.assertEqual(_resolve_groq_model(), DEFAULT_GROQ_MODEL)

    def test_replaces_deprecated_groq_prefixed_model(self) -> None:
        with patch.dict(os.environ, {"GROQ_MODEL": "groq/llama-3.1-8b-instant"}):
            self.assertEqual(_resolve_groq_model(), DEFAULT_GROQ_MODEL)

    def test_preserves_supported_custom_model(self) -> None:
        with patch.dict(os.environ, {"GROQ_MODEL": "groq/openai/gpt-oss-120b"}):
            self.assertEqual(_resolve_groq_model(), "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
