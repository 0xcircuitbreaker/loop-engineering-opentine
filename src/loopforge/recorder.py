"""opentine integration layer for loop runs.

This file keeps the opentine-specific mechanics in one place:
- creating/maintaining the Run object
- recording loop thoughts/tools/models/errors with explicit parents
- safe persistence to .tine
- helper for forked branch run creation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4
import os

from opentine import Run, StepKind, RunStatus


@dataclass
class RecordedStep:
    kind: StepKind
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None = None
    parent_id: str | None = None
    duration: float = 0.0
    cost: float = 0.0


@dataclass
class LoopRecorder:
    """Small wrapper around opentine.Run with loop-oriented defaults."""

    goal: str
    context: str = ""
    run: Run = field(init=False)
    path: Path | None = None
    _last_step_id: str | None = None

    def __post_init__(self) -> None:
        safe_goal = (self.goal or "")[:500]
        self.run = Run(
            manifest={
                "kind": "loop-engineering",
                "goal": safe_goal,
                "resume": True,
                "replay": ["cache", "rerun"],
                "created_via": "loopforge",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "context": (self.context or "")[:2000],
                "version": "0.1.0",
            },
            status=RunStatus.running,
            user_prompt=safe_goal,
        )

    def record(self, step: RecordedStep) -> str:
        """Record one step and return its step id."""
        added = self.run.add_step(
            step.kind,
            inputs=step.inputs,
            outputs=step.outputs,
            parent_id=step.parent_id,
            duration=step.duration,
            cost=step.cost,
        )
        self._last_step_id = added.id
        return added.id

    def record_think(self, payload: dict[str, Any], *, parent_id: str | None = None) -> str:
        return self.record(
            RecordedStep(
                kind=StepKind.think,
                inputs={"event": "loop_think", "payload": payload},
                parent_id=parent_id or self._last_step_id,
            )
        )

    def record_model(self, payload: dict[str, Any], *, outputs: dict[str, Any], parent_id: str | None = None,
                     duration: float = 0.0, cost: float = 0.0) -> str:
        return self.record(
            RecordedStep(
                kind=StepKind.model,
                inputs={"event": "loop_model", "payload": payload},
                outputs=outputs,
                parent_id=parent_id or self._last_step_id,
                duration=duration,
                cost=cost,
            )
        )

    def record_done(self, summary: str, *, parent_id: str | None = None) -> str:
        self.run.status = RunStatus.completed
        return self.record(
            RecordedStep(
                kind=StepKind.done,
                inputs={"event": "loop_complete", "summary": str(summary)[:5000]},
                outputs={"done_at": datetime.now(timezone.utc).isoformat()},
                parent_id=parent_id or self._last_step_id,
            )
        )

    def record_error(self, error: Exception, *, parent_id: str | None = None) -> str:
        self.run.status = RunStatus.failed
        return self.record(
            RecordedStep(
                kind=StepKind.error,
                inputs={"event": "loop_error"},
                outputs={"error": repr(error)},
                parent_id=parent_id or self._last_step_id,
            )
        )

    def fork(self, from_step_id: str, branch_id: str) -> "LoopRecorder":
        """Create a new branch from an existing step.

        The returned loop recorder shares provenance for all ancestors but has
        its own tip and can continue independently.
        """
        forked = LoopRecorder(goal=self.goal, context=self.context)
        forked.run = self.run.fork(
            from_step_id=from_step_id,
            new_run_id=f"{self.run.run_id[:10]}-{branch_id}-{uuid4().hex[:6]}",
            branch=branch_id,
        )
        forked._last_step_id = from_step_id
        return forked

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            if self.path is None:
                base = Path(os.path.expanduser("~/.local/share/loopforge"))
                base.mkdir(parents=True, exist_ok=True)
                self.path = base / f"{self.run.run_id}.tine"
            path = self.path
        p = Path(path)
        self.run.save(p)
        return p


def make_recorder(goal: str, context: str = "") -> LoopRecorder:
    """Factory kept for readability and future extension."""
    return LoopRecorder(goal=goal, context=context)
