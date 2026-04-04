# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python Env Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PythonReviewAction, PythonReviewObservation


class PythonEnv(
    EnvClient[PythonReviewAction, PythonReviewObservation, State]
):
    """
    Client for the Python Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with PythonEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task.title)
        ...
        ...     result = client.step(PythonReviewAction(operation="request_hint"))
        ...     print(result.observation.feedback)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PythonEnv.from_docker_image("python_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(PythonReviewAction(operation="finalize"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PythonReviewAction) -> Dict[str, Any]:
        """
        Convert PythonReviewAction to a JSON-safe payload.

        Args:
            action: PythonReviewAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PythonReviewObservation]:
        """
        Parse server response into StepResult[PythonReviewObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with PythonReviewObservation
        """
        observation = PythonReviewObservation(**payload.get("observation", {}))

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
