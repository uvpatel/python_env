# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python code-review environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import (
        PythonEnvConfig,
        PythonReviewAction,
        PythonReviewObservation,
        ReviewFinding,
        Severity,
        TaskDescriptor,
        TaskEvaluation,
    )
except ImportError:
    from models import (
        PythonEnvConfig,
        PythonReviewAction,
        PythonReviewObservation,
        ReviewFinding,
        Severity,
        TaskDescriptor,
        TaskEvaluation,
    )


@dataclass(frozen=True)
class ReviewTask:
    """Internal task definition with hidden grading data."""

    descriptor: TaskDescriptor
    reference_findings: List[ReviewFinding]
    hint: str


SEVERITY_WEIGHTS: Dict[Severity, float] = {
    "critical": 1.0,
    "warning": 0.6,
    "info": 0.3,
}


def _build_task_bank() -> Dict[str, ReviewTask]:
    """Create the built-in benchmark task set."""

    return {
        "py-review-easy": ReviewTask(
            descriptor=TaskDescriptor(
                task_id="py-review-easy",
                difficulty="easy",
                title="Unchecked division in helper",
                objective="Find the correctness issue and recommend a safe fix.",
                code=(
                    "def ratio(total, count):\n"
                    "    return total / count\n"
                ),
                max_steps=4,
                success_threshold=0.7,
            ),
            reference_findings=[
                ReviewFinding(
                    rule_id="divide-by-zero",
                    title="Division by zero is possible",
                    line=2,
                    category="bug",
                    severity="critical",
                    rationale="A count of zero raises ZeroDivisionError and crashes the caller.",
                    recommendation="Guard against zero before dividing or return a safe default.",
                )
            ],
            hint="Focus on which input values make the function raise before it returns.",
        ),
        "py-review-medium": ReviewTask(
            descriptor=TaskDescriptor(
                task_id="py-review-medium",
                difficulty="medium",
                title="Mutable default in accumulator",
                objective="Find the state-sharing bug and explain its effect across calls.",
                code=(
                    "def append_event(event, bucket=[]):\n"
                    "    bucket.append(event)\n"
                    "    return bucket\n"
                ),
                max_steps=4,
                success_threshold=0.7,
            ),
            reference_findings=[
                ReviewFinding(
                    rule_id="mutable-default",
                    title="Mutable default argument leaks state",
                    line=1,
                    category="bug",
                    severity="warning",
                    rationale="The same list instance is reused across calls, so unrelated callers share data.",
                    recommendation="Use None as the default and allocate a new list inside the function.",
                )
            ],
            hint="Check whether function-local state is actually recreated on each call.",
        ),
        "py-review-hard": ReviewTask(
            descriptor=TaskDescriptor(
                task_id="py-review-hard",
                difficulty="hard",
                title="Shell command built from user input",
                objective="Identify the security issue and recommend a safer implementation.",
                code=(
                    "import os\n\n"
                    "def show_file(name):\n"
                    "    os.system(f'type {name}')\n"
                ),
                max_steps=4,
                success_threshold=0.75,
            ),
            reference_findings=[
                ReviewFinding(
                    rule_id="shell-injection",
                    title="Unsanitized shell command execution",
                    line=4,
                    category="security",
                    severity="critical",
                    rationale="User-controlled input reaches a shell command, enabling command injection.",
                    recommendation="Avoid shell execution and open the file directly with Python APIs.",
                )
            ],
            hint="Trace whether external input is executed by a shell instead of handled as data.",
        ),
    }


class PythonEnvironment(Environment[PythonReviewAction, PythonReviewObservation, State]):
    """Deterministic code-review environment used by the benchmark server."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Optional[PythonEnvConfig] = None):
        super().__init__()
        self._config = config or PythonEnvConfig()
        self._task_bank = _build_task_bank()
        self._task_cursor = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[ReviewTask] = None
        self._submitted_findings: List[ReviewFinding] = []
        self._hints_used = 0
        self._last_score = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> PythonReviewObservation:
        """Start a fresh episode on the next configured task."""

        del seed, kwargs
        task_id = self._config.task_order[self._task_cursor % len(self._config.task_order)]
        self._task_cursor += 1
        self._current_task = self._task_bank.get(task_id) or next(iter(self._task_bank.values()))
        self._submitted_findings = []
        self._hints_used = 0
        self._last_score = 0.0
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._current_task.descriptor.task_id,
        )
        return self._build_observation(
            feedback="Inspect the code and submit structured findings.",
            evaluation=TaskEvaluation(total_findings=len(self._current_task.reference_findings)),
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: PythonReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> PythonReviewObservation:
        """Apply one review action and return updated benchmark state."""

        del timeout_s, kwargs
        if self._current_task is None:
            return self.reset()

        self._state.step_count += 1
        feedback = ""

        if action.operation in {"submit_findings", "finalize"}:
            self._submitted_findings.extend(action.findings)
            evaluation = self._evaluate(
                findings=self._submitted_findings,
                patched_code=action.patched_code,
            )
            feedback = (
                f"Matched {evaluation.matched_findings}/{evaluation.total_findings} reference findings"
                f" with {evaluation.false_positives} false positives."
            )
        else:
            self._hints_used += 1
            evaluation = self._evaluate(
                findings=self._submitted_findings,
                patched_code=None,
            )
            feedback = self._current_task.hint

        attempts_remaining = max(0, self._config.max_steps_per_task - self._state.step_count)
        done = action.operation == "finalize" or attempts_remaining == 0
        score_after_penalties = max(
            0.0,
            min(
                1.0,
                evaluation.score - (self._hints_used * self._config.hint_penalty),
            ),
        )
        evaluation = evaluation.model_copy(
            update={
                "score": score_after_penalties,
                "passed": score_after_penalties >= self._current_task.descriptor.success_threshold,
            }
        )
        reward = round(score_after_penalties - self._last_score, 4)
        self._last_score = score_after_penalties

        if done and evaluation.passed:
            feedback = f"{feedback} Task passed."
        elif done:
            feedback = f"{feedback} Task failed."

        return self._build_observation(
            feedback=feedback,
            evaluation=evaluation,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> State:
        """Return the current environment state."""

        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Expose metadata for the generated server docs."""

        return EnvironmentMetadata(
            name="python_env",
            description="A small deterministic Python code-review benchmark.",
            version="0.1.0",
        )

    def _build_observation(
        self,
        *,
        feedback: str,
        evaluation: TaskEvaluation,
        reward: float,
        done: bool,
    ) -> PythonReviewObservation:
        """Construct the observation payload returned to clients."""

        if self._current_task is None:
            raise RuntimeError("Environment has no active task.")

        attempts_remaining = max(0, self._config.max_steps_per_task - self._state.step_count)
        return PythonReviewObservation(
            task=self._current_task.descriptor,
            instructions="Inspect the code, submit findings, and finalize when done.",
            feedback=feedback,
            submitted_findings=list(self._submitted_findings),
            hints_used=self._hints_used,
            attempts_remaining=attempts_remaining,
            evaluation=evaluation,
            score=evaluation.score,
            review_time_ms=float(self._state.step_count * 100),
            done=done,
            reward=reward,
            metadata={
                "task_id": self._current_task.descriptor.task_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def _evaluate(
        self,
        *,
        findings: List[ReviewFinding],
        patched_code: Optional[str],
    ) -> TaskEvaluation:
        """Score submitted findings against the hidden task references."""

        if self._current_task is None:
            return TaskEvaluation()

        reference_findings = self._current_task.reference_findings
        matched_reference_ids: List[str] = []
        duplicate_findings = 0
        false_positives = 0

        for finding in findings:
            reference = self._match_reference(finding, reference_findings)
            if reference is None:
                false_positives += 1
                continue

            reference_id = reference.rule_id or f"line-{reference.line}"
            if reference_id in matched_reference_ids:
                duplicate_findings += 1
            else:
                matched_reference_ids.append(reference_id)

        total_weight = sum(SEVERITY_WEIGHTS[item.severity] for item in reference_findings)
        matched_weight = sum(
            SEVERITY_WEIGHTS[item.severity]
            for item in reference_findings
            if (item.rule_id or f"line-{item.line}") in matched_reference_ids
        )
        weighted_recall = matched_weight / total_weight if total_weight else 1.0
        patch_score = (
            1.0
            if patched_code and patched_code.strip() and patched_code.strip() != self._current_task.descriptor.code.strip()
            else 0.0
        )
        score = weighted_recall
        score -= false_positives * self._config.false_positive_penalty
        score -= duplicate_findings * self._config.duplicate_penalty
        score += patch_score * self._config.patch_bonus_multiplier
        score = max(0.0, min(1.0, score))

        return TaskEvaluation(
            matched_reference_ids=matched_reference_ids,
            matched_findings=len(matched_reference_ids),
            total_findings=len(reference_findings),
            false_positives=false_positives,
            duplicate_findings=duplicate_findings,
            weighted_recall=round(weighted_recall, 4),
            patch_score=patch_score,
            score=round(score, 4),
            passed=score >= self._current_task.descriptor.success_threshold,
        )

    def _match_reference(
        self,
        finding: ReviewFinding,
        references: List[ReviewFinding],
    ) -> Optional[ReviewFinding]:
        """Match a submitted finding against one hidden reference finding."""

        normalized_title = finding.title.strip().lower()
        for reference in references:
            if finding.rule_id and reference.rule_id == finding.rule_id:
                return reference
            if finding.line == reference.line and finding.category == reference.category:
                return reference
            if normalized_title and normalized_title == reference.title.strip().lower():
                return reference
        return None
