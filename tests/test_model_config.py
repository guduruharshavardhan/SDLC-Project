import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.crew_pipeline import DEFAULT_GROQ_MODEL, _resolve_groq_model, run_sdlc_pipeline


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



class PipelineEngineTests(unittest.TestCase):
    def _run_pipeline(self, modules: dict[str, object], crewai_error: Exception | None = None) -> dict:
        tracker = MagicMock()
        tracker.run = SimpleNamespace(run_id="test-run")
        tracker.summary.return_value = {}
        artifacts = {name: "result" for name in (
            "Business Analyst",
            "Solution Architect",
            "Full-Stack Developer",
            "QA Engineer",
            "Technical Writer",
        )}
        with (
            patch.dict(sys.modules, modules),
            patch("agents.crew_pipeline.build_context", return_value="context"),
            patch("agents.crew_pipeline.ObservabilityTracker", return_value=tracker),
            patch("agents.crew_pipeline.write_release_notes"),
            patch("agents.crew_pipeline._run_with_groq", return_value=artifacts),
            patch(
                "agents.crew_pipeline._run_with_crewai",
                side_effect=crewai_error or AssertionError("CrewAI runner should not be called"),
            ),
        ):
            result = run_sdlc_pipeline("requirement")
        return result

    def test_missing_optional_crewai_uses_clean_groq_engine_label(self) -> None:
        result = self._run_pipeline({"crewai": None})
        self.assertTrue(result["ok"])
        self.assertEqual(result["engine"], "groq-sequential")

    def test_crewai_runtime_failure_remains_visible_in_engine_label(self) -> None:
        result = self._run_pipeline(
            {"crewai": types.ModuleType("crewai")},
            crewai_error=RuntimeError("crew failed"),
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            result["engine"],
            "groq-sequential (crewai_error=RuntimeError: crew failed)",
        )
if __name__ == "__main__":
    unittest.main()
