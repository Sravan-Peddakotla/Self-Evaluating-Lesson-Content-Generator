"""Provider adapters, strict lesson evaluation, and durable run memory."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUBRIC = (
    ("accuracy", "Accurate and grounded", "Core claims are accurate, appropriately qualified, and do not invent facts or sources."),
    ("beginner_language", "Beginner-friendly language", "A learner starting from zero can understand the explanations; sentences and vocabulary stay simple."),
    ("example", "Concrete worked example", "At least one concrete example walks through what happens and why."),
    ("jargon", "No unexplained jargon", "Technical terms are explained in plain language before or when they are used."),
    ("coverage", "Complete topic coverage", "Explains what the topic is, why it matters, and how it works."),
    ("flow", "Coherent teaching flow", "Ideas build in a sensible order and the lesson stands alone."),
)
_RETRYABLE_GEMINI_CODES = {408, 429, 500, 502, 503, 504}
_MAX_GEMINI_RETRIES = 2


def call_model(provider: str, model: str, api_key: str, prompt: str) -> str:
    """Call one supported text-generation API and return its text response."""
    if provider == "Google Gemini":
        from google import genai

        for retry_index in range(_MAX_GEMINI_RETRIES + 1):
            client = genai.Client(api_key=api_key)
            retry_delay = None
            try:
                response = client.interactions.create(
                    model=model,
                    input=prompt,
                )
                text = response.output_text
            except genai.errors.APIError as error:
                if error.code not in _RETRYABLE_GEMINI_CODES or retry_index == _MAX_GEMINI_RETRIES:
                    raise
                retry_delay = 2**retry_index
            finally:
                client.close()

            if retry_delay is None:
                break
            time.sleep(retry_delay)
    elif provider == "OpenAI":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )
            text = response.output_text
        finally:
            client.close()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not text or not text.strip():
        raise RuntimeError("The model returned an empty response.")
    return text.strip()


def _json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences and short preambles."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator response did not contain a JSON object.")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Evaluator response must be a JSON object.")
    return parsed


def build_generation_prompt(
    topic: str,
    rejection_reasons: list[str],
    adaptive_guidance: str,
) -> str:
    retry_context = ""
    if rejection_reasons:
        retry_context = "\n\nREQUIRED FIXES FROM THE LAST EVALUATION:\n" + "\n".join(
            f"- {reason}" for reason in rejection_reasons
        )
    memory_context = (
        f"\n\nLESSONS FROM PREVIOUS RUNS:\n{adaptive_guidance}"
        if adaptive_guidance
        else ""
    )
    return f"""You are a careful teacher creating a standalone beginner lesson.
Learner: a 12th-grade graduate in India with limited English vocabulary and no prior knowledge of the topic.
Topic: {topic}

Teach what it is, why it matters, and how it works. Use short sentences and common words. Explain every technical term in plain language. Include a specific, worked example that follows the process step by step. Organize the lesson with clear Markdown headings and a short recap. Be factually careful: qualify uncertainty and do not invent citations, studies, or product claims. Do not mention this prompt or the evaluation process.
{retry_context}{memory_context}
"""


def build_evaluation_prompt(topic: str, lesson: str) -> str:
    checks = "\n".join(
        f'- "{key}": {label}. {description}'
        for key, label, description in RUBRIC
    )
    return f"""Act as a strict, independent quality reviewer. Evaluate only the lesson below for the topic {topic!r} and the stated beginner audience.

For each checkpoint, return pass=true only when the entire checkpoint is clearly satisfied. Otherwise pass=false and give a specific, actionable reason. Do not award partial credit. Check factual claims rather than assuming they are correct. The lesson must stand alone.

CHECKPOINTS:
{checks}

Return only a JSON object in this exact shape:
{{"checks":[{{"id":"accuracy","pass":false,"reason":"specific reason"}}],"retry_guidance":["specific change to make"]}}
Include every checkpoint exactly once. If a checkpoint passes, its reason can be an empty string.

LESSON TO REVIEW:
<lesson>
{lesson}
</lesson>
"""


def evaluate_lesson(provider: str, model: str, api_key: str, topic: str, lesson: str) -> dict[str, Any]:
    response = call_model(provider, model, api_key, build_evaluation_prompt(topic, lesson))
    parsed = _json_object(response)
    raw_checks = parsed.get("checks", [])
    by_id = {
        item.get("id"): item
        for item in raw_checks
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(raw_checks, list) else {}

    checks = []
    for key, label, _ in RUBRIC:
        item = by_id.get(key)
        passed = isinstance(item, dict) and item.get("pass") is True
        reason = item.get("reason", "") if isinstance(item, dict) else "Evaluator omitted this checkpoint."
        if not passed and not reason:
            reason = "Checkpoint did not receive an explicit pass."
        checks.append({"id": key, "label": label, "pass": passed, "reason": str(reason)})

    guidance = parsed.get("retry_guidance", [])
    if not isinstance(guidance, list):
        guidance = []
    return {
        "checks": checks,
        "passed": all(item["pass"] for item in checks),
        "retry_guidance": [str(item) for item in guidance if item],
    }


def run_pipeline(
    provider: str,
    model: str,
    api_key: str,
    topic: str,
    max_retries: int,
    adaptive_guidance: str = "",
) -> dict[str, Any]:
    """Generate, judge, and retry until every hard checkpoint passes or retries run out."""
    attempts = []
    rejection_reasons: list[str] = []
    lesson = ""
    evaluation: dict[str, Any] = {"checks": [], "passed": False, "retry_guidance": []}

    for attempt_number in range(max_retries + 1):
        lesson = call_model(
            provider,
            model,
            api_key,
            build_generation_prompt(topic, rejection_reasons, adaptive_guidance),
        )
        try:
            evaluation = evaluate_lesson(provider, model, api_key, topic, lesson)
        except (ValueError, json.JSONDecodeError) as error:
            evaluation = {
                "checks": [
                    {
                        "id": key,
                        "label": label,
                        "pass": False,
                        "reason": f"The evaluator response could not be read ({error}). Recheck this requirement in the next draft.",
                    }
                    for key, label, _ in RUBRIC
                ],
                "passed": False,
                "retry_guidance": ["Return a complete, valid evaluation with every rubric checkpoint."],
            }
        attempts.append({
            "attempt": attempt_number + 1,
            "lesson": lesson,
            "evaluation": evaluation,
        })
        if evaluation["passed"]:
            break
        rejection_reasons = [
            f"{check['label']}: {check['reason']}"
            for check in evaluation["checks"]
            if not check["pass"]
        ]
        rejection_reasons.extend(evaluation["retry_guidance"])

    return {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "provider": provider,
        "model": model,
        "passed": evaluation["passed"],
        "lesson": lesson,
        "attempts": attempts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class LessonMemory:
    """SQLite-backed run history, recurring failure patterns, and human feedback."""

    def __init__(self, database_path: str | Path | None = None):
        default_path = Path(__file__).with_name("lesson_memory.sqlite3")
        self.database_path = Path(database_path or os.getenv("LESSON_MEMORY_DB", default_path))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    lesson TEXT NOT NULL,
                    attempts_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    rating TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def save_run(self, run: dict[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run["id"], run["topic"], run["provider"], run["model"],
                    int(run["passed"]), run["lesson"], json.dumps(run["attempts"]),
                    run["created_at"],
                ),
            )

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id, topic, provider, model, passed, created_at FROM runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["passed"] = bool(result["passed"])
        result["attempts"] = json.loads(result.pop("attempts_json"))
        return result

    def add_feedback(self, run_id: str, rating: str, note: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO feedback (run_id, rating, note, created_at) VALUES (?, ?, ?, ?)",
                (run_id, rating, note.strip(), datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    def feedback_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT rating, note, created_at FROM feedback WHERE run_id = ? ORDER BY id DESC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def adaptive_guidance(self) -> str:
        """Turn recurring rubric failures into concise checks for the next generation."""
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT attempts_json FROM runs ORDER BY created_at DESC LIMIT 50").fetchall()
        counts: dict[str, int] = {}
        labels: dict[str, str] = {}
        for row in rows:
            failed_in_run = set()
            for attempt in json.loads(row["attempts_json"]):
                for check in attempt["evaluation"].get("checks", []):
                    if not check.get("pass"):
                        key = check["id"]
                        failed_in_run.add(key)
                        labels[key] = check["label"]
            for key in failed_in_run:
                counts[key] = counts.get(key, 0) + 1
        recurring = sorted(
            (key for key, count in counts.items() if count >= 2),
            key=lambda key: counts[key],
            reverse=True,
        )[:3]
        guidance = ""
        if recurring:
            guidance = "Previously recurring weak points: " + "; ".join(
                f"{labels[key]} (failed {counts[key]} times). Check this carefully before finishing."
                for key in recurring
            )
        with closing(self._connect()) as connection, connection:
            feedback_rows = connection.execute(
                "SELECT rating, note FROM feedback WHERE trim(note) != '' ORDER BY id DESC LIMIT 5"
            ).fetchall()
        feedback = [f"{row['rating']}: {row['note'][:300]}" for row in feedback_rows]
        if feedback:
            guidance += ("\n" if guidance else "") + "Human feedback from earlier runs: " + " | ".join(feedback)
        return guidance