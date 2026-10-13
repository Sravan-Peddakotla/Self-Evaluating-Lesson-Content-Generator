import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import errors as gemini_errors

from lesson_engine import LessonMemory, RUBRIC, _json_object, call_model, run_pipeline


class LessonEngineTests(unittest.TestCase):
    def test_gemini_client_stays_open_until_response_returns(self):
        client = Mock()
        client.interactions.create.return_value = SimpleNamespace(output_text="lesson")
        with patch("google.genai.Client", return_value=client):
            result = call_model("Google Gemini", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        self.assertEqual(
            client.mock_calls,
            [
                unittest.mock.call.interactions.create(model="test-model", input="prompt"),
                unittest.mock.call.close(),
            ],
        )

    def test_gemini_retries_temporary_503_and_closes_each_client(self):
        first_client = Mock()
        first_client.interactions.create.side_effect = gemini_errors.ServerError(
            503,
            {"error": {"message": "temporarily busy"}},
        )
        second_client = Mock()
        second_client.interactions.create.return_value = SimpleNamespace(output_text="lesson")
        with (
            patch("google.genai.Client", side_effect=[first_client, second_client]),
            patch("lesson_engine.time.sleep") as sleep,
        ):
            result = call_model("Google Gemini", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        first_client.close.assert_called_once()
        second_client.close.assert_called_once()
        sleep.assert_called_once_with(1)

    def test_gemini_does_not_retry_permanent_api_errors(self):
        client = Mock()
        client.interactions.create.side_effect = gemini_errors.ClientError(
            404,
            {"error": {"message": "model not found"}},
        )
        with (
            patch("google.genai.Client", return_value=client),
            patch("lesson_engine.time.sleep") as sleep,
            self.assertRaises(gemini_errors.ClientError),
        ):
            call_model("Google Gemini", "missing-model", "fake-key", "prompt")

        client.interactions.create.assert_called_once()
        client.close.assert_called_once()
        sleep.assert_not_called()

    def test_openai_client_stays_open_until_response_returns(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(output_text="lesson")
        with patch("openai.OpenAI", return_value=client):
            result = call_model("OpenAI", "test-model", "fake-key", "prompt")

        self.assertEqual(result, "lesson")
        self.assertEqual(
            client.mock_calls,
            [
                unittest.mock.call.responses.create(model="test-model", input="prompt"),
                unittest.mock.call.close(),
            ],
        )

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