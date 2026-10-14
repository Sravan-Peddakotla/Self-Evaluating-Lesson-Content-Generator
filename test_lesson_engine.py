import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lesson_engine import LessonMemory, RUBRIC, _json_object, call_model, run_pipeline


class LessonEngineTests(unittest.TestCase):
    def test_gemini_langchain_model_returns_response_text(self):
        chat_model = Mock()
        chat_model.invoke.return_value = SimpleNamespace(text="lesson")
        with patch("lesson_engine.ChatGoogleGenerativeAI", return_value=chat_model) as model_factory:
            result = call_model("Google Gemini", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        model_factory.assert_called_once_with(model="test-model", api_key="fake-key", retries=0)
        chat_model.invoke.assert_called_once_with("prompt")

    def test_gemini_retries_temporary_503_and_closes_each_client(self):
        first_model = Mock()
        first_model.invoke.side_effect = RuntimeError("temporarily busy")
        first_model.invoke.side_effect.code = 503
        second_model = Mock()
        second_model.invoke.return_value = SimpleNamespace(text="lesson")
        with (
            patch("lesson_engine.ChatGoogleGenerativeAI", side_effect=[first_model, second_model]),
            patch("lesson_engine.time.sleep") as sleep,
        ):
            result = call_model("Google Gemini", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        first_model.invoke.assert_called_once_with("prompt")
        second_model.invoke.assert_called_once_with("prompt")
        sleep.assert_called_once_with(1)

    def test_gemini_does_not_retry_permanent_api_errors(self):
        chat_model = Mock()
        model_error = RuntimeError("model not found")
        model_error.code = 404
        chat_model.invoke.side_effect = model_error
        with (
            patch("lesson_engine.ChatGoogleGenerativeAI", return_value=chat_model),
            patch("lesson_engine.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "model not found"),
        ):
            call_model("Google Gemini", "missing-model", "fake-key", "prompt")

        chat_model.invoke.assert_called_once_with("prompt")
        sleep.assert_not_called()

    def test_openai_langchain_model_returns_response_text(self):
        chat_model = Mock()
        chat_model.invoke.return_value = SimpleNamespace(text="lesson")
        with patch("lesson_engine.ChatOpenAI", return_value=chat_model) as model_factory:
            result = call_model("OpenAI", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        model_factory.assert_called_once_with(model="test-model", api_key="fake-key")
        chat_model.invoke.assert_called_once_with("prompt")

    def test_groq_langchain_model_returns_response_text(self):
        chat_model = Mock()
        chat_model.invoke.return_value = SimpleNamespace(text="lesson")
        with patch("lesson_engine.ChatGroq", return_value=chat_model) as model_factory:
            result = call_model("Groq", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        model_factory.assert_called_once_with(
            model="test-model",
            api_key="fake-key",
            max_retries=2,
        )
        chat_model.invoke.assert_called_once_with("prompt")

    def test_json_parser_accepts_fenced_response(self):
        self.assertEqual(_json_object('```json\n{"ok": true}\n```'), {"ok": True})

    def test_retry_receives_failed_checkpoint_reasons(self):
        failed = {
            "checks": [
                {"id": key, "pass": key != "example", "reason": "Needs a concrete walkthrough."}
                for key, _, _ in RUBRIC
            ],
            "retry_guidance": [],
        }
        passed = {
            "checks": [
                {"id": key, "pass": True, "reason": ""}
                for key, _, _ in RUBRIC
            ],
            "retry_guidance": [],
        }
        with patch("lesson_engine.call_model", side_effect=[
            "First draft", json.dumps(failed), "Revised draft", json.dumps(passed),
        ]) as model_call:
            result = run_pipeline("OpenAI", "test-model", "fake-key", "Introduction to RAG", 1)

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["attempts"]), 2)
        retry_prompt = model_call.call_args_list[2].args[3]
        self.assertIn("Needs a concrete walkthrough", retry_prompt)

    def test_progress_reports_three_retries_and_stops_after_four_drafts(self):
        failed = {
            "checks": [
                {"id": key, "pass": False, "reason": "Needs improvement."}
                for key, _, _ in RUBRIC
            ],
            "retry_guidance": [],
        }
        progress = []
        with patch("lesson_engine.call_model", side_effect=[
            "Draft one", json.dumps(failed),
            "Draft two", json.dumps(failed),
            "Draft three", json.dumps(failed),
            "Draft four", json.dumps(failed),
        ]):
            result = run_pipeline(
                "Groq", "test-model", "fake-key", "Introduction to RAG", 3,
                progress_callback=progress.append,
            )

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["attempts"]), 4)
        self.assertTrue(any("Regenerating lesson draft 2" in item for item in progress))
        self.assertTrue(any("Regenerating lesson draft 3" in item for item in progress))
        self.assertTrue(any("Regenerating lesson draft 4" in item for item in progress))
        self.assertEqual(progress[-1], "Failed to pass the rubric after 4 drafts.")

    def test_retry_limit_cannot_exceed_three(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            run_pipeline("Groq", "test-model", "fake-key", "RAG", 4)

    def test_recurring_failures_and_human_feedback_shape_future_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = LessonMemory(Path(temp_dir) / "memory.sqlite3")
            failed_check = {
                "id": "example",
                "label": "Concrete worked example",
                "pass": False,
                "reason": "Add a step-by-step example.",
            }
            attempt = {"evaluation": {"checks": [failed_check]}}
            run = {
                "id": "test-run",
                "topic": "RAG",
                "provider": "OpenAI",
                "model": "test-model",
                "passed": False,
                "lesson": "Draft",
                "attempts": [attempt, attempt],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            memory.save_run(run)
            second_run = {**run, "id": "second-test-run", "attempts": [attempt]}
            memory.save_run(second_run)
            memory.add_feedback(run["id"], "Needs work", "Use shorter sentences.")

            guidance = memory.adaptive_guidance()
            saved_run = memory.get_run(run["id"])

        self.assertIn("Concrete worked example", guidance)
        self.assertIn("failed 2 times", guidance)
        self.assertIn("Use shorter sentences", guidance)
        self.assertEqual(saved_run["attempts"], [attempt, attempt])


if __name__ == "__main__":
    unittest.main()