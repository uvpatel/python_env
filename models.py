"""Typed models for the Python code-review environment.

This module is the shared contract between:

- the OpenEnv server implementation
- the REST API layer
- the benchmark grader
- the inference script
- the tests

Keeping these models centralized makes the environment easier to validate,
serialize, and evolve without each module inventing its own payload shape.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


# Difficulty buckets are intentionally small and fixed so tasks can be
# grouped for curriculum learning and reporting without extra normalization.
Difficulty = Literal["easy", "medium", "hard"]

# Severity is separate from category because one category such as "security"
# can still vary in importance across tasks.
Severity = Literal["critical", "warning", "info"]

# Categories help both humans and agents understand what type of issue was found.
Category = Literal["bug", "security", "style", "performance", "maintainability"]

# Operations define the small action space an agent can use during an episode.
Operation = Literal["submit_findings", "request_hint", "finalize"]


class ReviewFinding(BaseModel):
    """A structured review finding.

    Each finding is designed to be machine-gradable while still resembling the
    sort of issue summary a human reviewer would write in a real code review.
    """

    title: str = Field(..., description="Short title for the finding")
    line: Optional[int] = Field(default=None, description="1-based source line number")
    category: Category = Field(default="bug", description="Issue category")
    severity: Severity = Field(default="warning", description="Issue severity")
    rationale: str = Field(
        default="",
        description="Why the issue matters and how it affects behaviour or safety",
    )
    recommendation: Optional[str] = Field(
        default=None, description="Concrete fix recommendation"
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Stable internal rule identifier when known",
    )


class TaskDescriptor(BaseModel):
    """Public task metadata shown to the agent.

    This is intentionally the "visible" task information. Hidden grading
    details stay inside the server task bank so the benchmark remains useful.
    """

    task_id: str = Field(..., description="Stable task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty bucket")
    title: str = Field(..., description="Short task title")
    objective: str = Field(..., description="What the reviewer should accomplish")
    code: str = Field(..., description="Python code to review")
    max_steps: int = Field(..., ge=1, description="Maximum actions allowed")
    success_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Minimum score considered a pass"
    )


class TaskEvaluation(BaseModel):
    """Deterministic grader output.

    This model is returned in observations and offline grading routes so that
    both online interaction and offline evaluation use exactly the same metrics.
    """

    matched_reference_ids: List[str] = Field(default_factory=list)
    matched_findings: int = Field(default=0, ge=0)
    total_findings: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    duplicate_findings: int = Field(default=0, ge=0)
    weighted_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = Field(default=False)


class PythonReviewAction(Action):
    """Action submitted by an agent during an episode.

    The action space is kept intentionally small:

    - `submit_findings` for intermediate progress
    - `request_hint` when the agent needs guidance at a small penalty
    - `finalize` when the agent wants the episode to end
    """

    operation: Operation = Field(
        default="submit_findings",
        description="How to interact with the environment on this step",
    )
    findings: List[ReviewFinding] = Field(
        default_factory=list,
        description="Structured findings being submitted for grading",
    )
    patched_code: Optional[str] = Field(
        default=None,
        description="Optional improved version of the code under review",
    )
    note: Optional[str] = Field(
        default=None,
        description="Optional free-form reviewer note for logging or context",
    )


class PythonEnvConfig(BaseModel):
    """Environment-level configuration knobs.

    These values are useful for experimentation because they let you adjust
    reward shaping and curriculum ordering without changing the grader logic.
    """

    task_order: List[str] = Field(
        default_factory=lambda: ["py-review-easy", "py-review-medium", "py-review-hard"],
        description="Deterministic task order used across resets",
    )
    max_steps_per_task: int = Field(default=4, ge=1, le=10)
    hint_penalty: float = Field(default=0.05, ge=0.0, le=1.0)
    false_positive_penalty: float = Field(default=0.08, ge=0.0, le=1.0)
    duplicate_penalty: float = Field(default=0.03, ge=0.0, le=1.0)
    patch_bonus_multiplier: float = Field(default=0.2, ge=0.0, le=1.0)
    max_history_entries: int = Field(default=50, ge=1, le=500)


class PythonReviewObservation(Observation):
    """Observation returned by `reset()` and `step()`.

    The observation combines:

    - visible task context
    - immediate feedback on the previous action
    - cumulative evaluation state
    - OpenEnv-standard reward/done/metadata fields
    """

    task: TaskDescriptor = Field(..., description="Current task details")
    instructions: str = Field(
        default="Inspect the code and submit structured findings.",
        description="Episode instructions shown to the agent",
    )
    feedback: str = Field(default="", description="Feedback for the last action")
    submitted_findings: List[ReviewFinding] = Field(
        default_factory=list,
        description="All findings submitted so far in this episode",
    )
    hints_used: int = Field(default=0, ge=0)
    attempts_remaining: int = Field(default=0, ge=0)
    evaluation: TaskEvaluation = Field(default_factory=TaskEvaluation)
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current task score after this step",
    )
    review_time_ms: float = Field(default=0.0, ge=0.0)


class EpisodeRecord(BaseModel):
    """Stored summary of a completed or in-progress episode.

    This model is used by the custom history routes and is intentionally
    compact enough to archive for later analysis or dataset creation.
    """

    episode_id: str
    task_id: str
    difficulty: Difficulty
    title: str
    final_score: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(default=False)
    steps_taken: int = Field(default=0, ge=0)
    hints_used: int = Field(default=0, ge=0)
    matched_findings: int = Field(default=0, ge=0)
    total_findings: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    duplicate_findings: int = Field(default=0, ge=0)
    status: Literal["active", "completed"] = Field(default="completed")
    created_at: str
    updated_at: str


class DirectReviewRequest(BaseModel):
    """Request model for ad-hoc review outside the benchmark tasks."""

    code: str = Field(..., description="Python source code to inspect")
    context: Optional[str] = Field(
        default=None, description="Optional explanation of the code's purpose"
    )


class DirectReviewResponse(BaseModel):
    """Static review result for arbitrary Python code.

    This route is useful for manual testing and dataset generation because it
    lets you review arbitrary snippets without entering the benchmark loop.
    """

    issues: List[ReviewFinding] = Field(default_factory=list)
    summary: str = Field(default="")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    improved_code: Optional[str] = Field(default=None)


class DeleteResponse(BaseModel):
    """Small acknowledgement payload for DELETE routes."""

    detail: str


class HealthResponse(BaseModel):
    """Health payload used by Docker and Spaces checks.

    This payload stays intentionally simple because health checks are often
    consumed by infrastructure rather than by human users.
    """

    status: Literal["ok"] = "ok"
    environment: str = "python_env"
    task_count: int = Field(default=0, ge=0)
    active_task_id: Optional[str] = None
    active_episode_id: Optional[str] = None


# Backward-compatible aliases keep older imports working while the project
# standardizes on the `Python*` naming convention.
PythonAction = PythonReviewAction
PythonObservation = PythonReviewObservation
CodeReviewAction = PythonReviewAction
CodeReviewObservation = PythonReviewObservation
CodeReviewConfig = PythonEnvConfig
