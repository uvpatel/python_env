"""Baseline inference script for the Python code-review environment.

This script is meant to be submission-friendly:

- configuration comes from environment variables
- model calls use the OpenAI client as required
- malformed model output is handled gracefully
- a JSON report is written for reproducibility
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import PythonEnv
from models import PythonReviewAction, ReviewFinding


# Read all runtime configuration from environment variables so the script can
# be reused unchanged across local runs, CI, and HF Spaces validation.
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
DOCKER_IMAGE = os.getenv("PYTHON_ENV_IMAGE", "python_env-env:latest")
MAX_STEPS = int(os.getenv("MAX_STEPS", "3"))
MAX_TASKS = int(os.getenv("MAX_TASKS", "3"))
REPORT_PATH = Path(os.getenv("INFERENCE_REPORT_PATH", "inference_results.json"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))

SYSTEM_PROMPT = """You are a precise Python code reviewer.
Return strict JSON using this schema:
{
  "findings": [
    {
      "title": "short title",
      "line": 1,
      "category": "bug|security|style|performance|maintainability",
      "severity": "critical|warning|info",
      "rationale": "why it matters",
      "recommendation": "how to fix it",
      "rule_id": "optional-stable-id"
    }
  ],
  "patched_code": null
}

Rules:
- Output JSON only. No markdown fences.
- Only report issues supported by the visible code.
- Prefer high precision over quantity.
- Include line numbers when possible.
"""


def _build_prompt(observation, step: int, history: List[str]) -> str:
    """Build the task prompt sent to the model for one step."""

    history_text = "\n".join(history[-4:]) if history else "No previous attempts."
    return (
        f"Task ID: {observation.task.task_id}\n"
        f"Difficulty: {observation.task.difficulty}\n"
        f"Objective: {observation.task.objective}\n"
        f"Step: {step}\n"
        f"Attempts remaining: {observation.attempts_remaining}\n"
        f"Current score: {observation.score:.2f}\n"
        f"Latest feedback: {observation.feedback or 'None'}\n"
        f"Attempt history:\n{history_text}\n\n"
        "Code to review:\n"
        "```python\n"
        f"{observation.task.code}\n"
        "```"
    )


def _extract_text_content(message_content: Any) -> str:
    """Normalize OpenAI response content into one text string."""

    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_json_blob(content: str) -> str:
    """Extract a JSON object from plain or fenced model output."""

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def _parse_response(content: str) -> Dict[str, Any]:
    """Parse the model response into a normalized payload dict."""

    raw = _extract_json_blob(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"findings": [], "patched_code": None, "_parse_error": raw}

    findings = data.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    patched_code = data.get("patched_code")
    if patched_code is not None and not isinstance(patched_code, str):
        patched_code = None
    return {"findings": findings, "patched_code": patched_code}


def _completion(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """Send one completion request to the configured model endpoint."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = _extract_text_content(response.choices[0].message.content) or "{}"
    return _parse_response(content)


def _normalize_findings(payload: Dict[str, Any]) -> List[ReviewFinding]:
    """Convert raw dict findings into validated `ReviewFinding` objects."""

    findings: List[ReviewFinding] = []
    for item in payload.get("findings", []):
        if not isinstance(item, dict):
            continue
        try:
            findings.append(ReviewFinding(**item))
        except Exception:
            continue
    return findings


def _build_fallback_action(observation, note: str) -> PythonReviewAction:
    """Create a safe fallback action when model output is unusable."""

    return PythonReviewAction(
        operation="finalize" if observation.attempts_remaining <= 1 else "request_hint",
        note=note,
    )


def _to_action(
    payload: Dict[str, Any],
    observation,
    finalize: bool,
) -> PythonReviewAction:
    """Convert a parsed model payload into a valid environment action."""

    findings = _normalize_findings(payload)
    if not findings and not payload.get("patched_code"):
        note = "Model returned no valid findings."
        if payload.get("_parse_error"):
            note = f"{note} Raw response could not be parsed as JSON."
        return _build_fallback_action(observation, note)

    return PythonReviewAction(
        operation="finalize" if finalize else "submit_findings",
        findings=findings,
        patched_code=payload.get("patched_code"),
    )


def _make_env() -> PythonEnv:
    """Connect to a live environment or launch the Docker image."""

    if ENV_BASE_URL:
        return PythonEnv(base_url=ENV_BASE_URL)
    return PythonEnv.from_docker_image(DOCKER_IMAGE)


def _task_result_dict(observation, step_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the report payload for one completed task run."""

    evaluation = observation.evaluation
    return {
        "task_id": observation.task.task_id,
        "difficulty": observation.task.difficulty,
        "title": observation.task.title,
        "score": observation.score,
        "passed": evaluation.passed,
        "matched_findings": evaluation.matched_findings,
        "total_findings": evaluation.total_findings,
        "false_positives": evaluation.false_positives,
        "duplicate_findings": evaluation.duplicate_findings,
        "weighted_recall": evaluation.weighted_recall,
        "patch_score": evaluation.patch_score,
        "steps": step_logs,
    }


def main() -> None:
    """Run the configured model against the benchmark task set."""

    if not API_KEY:
        raise RuntimeError("Set HF_TOKEN or OPENAI_API_KEY before running inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = _make_env()
    episode_results: List[Dict[str, Any]] = []

    try:
        for index in range(MAX_TASKS):
            result = env.reset()
            observation = result.observation
            history: List[str] = []
            step_logs: List[Dict[str, Any]] = []

            print(
                f"Task {index + 1}: {observation.task.task_id} "
                f"({observation.task.difficulty})"
            )

            for step in range(1, MAX_STEPS + 1):
                prompt = _build_prompt(observation, step, history)
                try:
                    # Model-call failures are captured in the report rather than
                    # crashing the full benchmark run.
                    payload = _completion(client, prompt)
                except Exception as exc:
                    payload = {"findings": [], "patched_code": None, "_error": str(exc)}

                action = _to_action(
                    payload=payload,
                    observation=observation,
                    finalize=step == MAX_STEPS or observation.attempts_remaining <= 1,
                )

                result = env.step(action)
                observation = result.observation

                step_log = {
                    "step": step,
                    "operation": action.operation,
                    "submitted_findings": len(action.findings),
                    "reward": result.reward or 0.0,
                    "score": observation.score,
                    "done": result.done,
                    "feedback": observation.feedback,
                }
                if payload.get("_error"):
                    step_log["model_error"] = payload["_error"]
                if payload.get("_parse_error"):
                    step_log["parse_error"] = True
                step_logs.append(step_log)

                # The history string is fed back into later prompts so the
                # model can see what it already tried.
                history.append(
                    f"step={step} op={action.operation} findings={len(action.findings)} "
                    f"score={observation.score:.2f} feedback={observation.feedback}"
                )

                print(
                    f"  step={step} op={action.operation} findings={len(action.findings)} "
                    f"score={observation.score:.2f} reward={(result.reward or 0.0):.2f} "
                    f"done={result.done}"
                )

                if result.done:
                    break

            episode_results.append(_task_result_dict(observation, step_logs))
    finally:
        env.close()

    mean_score = (
        sum(item["score"] for item in episode_results) / len(episode_results)
        if episode_results
        else 0.0
    )
    summary = {
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "task_count": len(episode_results),
        "mean_score": mean_score,
        "results": episode_results,
    }

    # Persist the report so scores can be compared across runs and models.
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()