"""Loop execution policy helpers.

Policies are enforced by the engine and embedded in opentine run manifests so the
result artifact carries the policy context that governed branching and termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoopPolicy:
    """Execution policy for a loop run.

    Args:
        max_steps: Optional hard cap for total model steps across the run.
        max_total_cost: Optional spend ceiling.
        max_duration_seconds: Optional wall-clock cap.
        min_score: Optional early-stop score threshold.
    """

    max_steps: int | None = None
    max_total_cost: float | None = None
    max_duration_seconds: float | None = None
    min_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_total_cost": self.max_total_cost,
            "max_duration_seconds": self.max_duration_seconds,
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "LoopPolicy":
        if not data:
            return cls()
        return cls(
            max_steps=data.get("max_steps"),
            max_total_cost=data.get("max_total_cost"),
            max_duration_seconds=data.get("max_duration_seconds"),
            min_score=data.get("min_score"),
        )


class LoopPolicyViolation(RuntimeError):
    """Raised when a policy boundary is reached."""

