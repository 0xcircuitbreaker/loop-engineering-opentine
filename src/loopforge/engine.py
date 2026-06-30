"""Core loop engine.

The engine is intentionally generic:
- Provide a pure transition function (`step_fn`).
- Every loop cycle is recorded as an opentine `model` step.
- Branches are represented as queue entries and can be forked via opentine.

The objective is simple: make complex iterative loops observable and rewritable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .recorder import LoopRecorder


@dataclass
class LoopStepContext:
    iteration: int
    branch_id: str
    current_state: dict[str, Any]
    best_score: float | None
    history_len: int
    run_id: str


@dataclass
class LoopStepResult:
    """Output of one loop step.

    `next_states` lets one step fan out into multiple branches.
    """

    observation: str
    next_states: list[dict[str, Any]]
    score: float
    stop: bool = False
    model_outputs: dict[str, Any] | None = None
    duration: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopBranch:
    branch_id: str
    state: dict[str, Any]
    recorder: LoopRecorder
    last_step_id: str | None = None
    score: float = 0.0
    steps: int = 0
    best_state: dict[str, Any] | None = None


@dataclass
class LoopExecutionResult:
    best: LoopBranch
    all_branches: list[LoopBranch]
    best_step_result: LoopStepResult | None
    artifacts: list[Path]


class LoopFactory:
    """Small helper to create deterministic branch ids."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._counter = 0

    def branch_id(self, suffix: str = "") -> str:
        self._counter += 1
        if suffix:
            return f"{suffix}-b{self._counter}"
        return f"b{self._counter}"


class LoopEngine:
    """Generic engine for iterative, branchable loops with opentine-backed traces."""

    def __init__(
        self,
        step_fn: Callable[[LoopStepContext], LoopStepResult],
        *,
        max_steps: int = 40,
        max_branches: int = 6,
        branch_width: int = 3,
        target_score: float = float("inf"),
        autosave_steps: int = 0,
        runs_dir: str = "~/.local/share/loopforge",
    ) -> None:
        if max_branches < 1:
            raise ValueError("max_branches must be >= 1")
        if branch_width < 1:
            raise ValueError("branch_width must be >= 1")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.step_fn = step_fn
        self.max_steps = max_steps
        self.max_branches = max_branches
        self.branch_width = branch_width
        self.target_score = target_score
        self.autosave_steps = autosave_steps
        self.runs_dir = Path(runs_dir).expanduser()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, goal: str, initial_state: dict[str, Any], context: str = "") -> LoopExecutionResult:
        root = LoopBranch(
            branch_id="root",
            state=initial_state,
            recorder=LoopRecorder(goal=goal, context=context),
            score=float("-inf"),
            best_state=dict(initial_state),
        )
        root.recorder.record_think({"event": "loop_init", "state": initial_state})

        queue: deque[LoopBranch] = deque([root])
        all_seen: list[LoopBranch] = [root]
        best: LoopBranch | None = None
        best_step_result: LoopStepResult | None = None
        total_steps = 0

        branch_factory = LoopFactory(root.recorder.run.run_id)

        while queue and total_steps < self.max_steps:
            branch = queue.popleft()
            ctx = LoopStepContext(
                iteration=branch.steps + 1,
                branch_id=branch.branch_id,
                current_state=branch.state,
                best_score=best.score if best else None,
                history_len=branch.steps,
                run_id=branch.recorder.run.run_id,
            )

            try:
                step = self.step_fn(ctx)
            except Exception as exc:  # pragma: no cover - defensive path
                branch.recorder.record_error(exc, parent_id=branch.last_step_id)
                branch.recorder.save(self.runs_dir / f"{branch.recorder.run.run_id}.tine")
                continue

            step_id = branch.recorder.record_model(
                payload={
                    "event": "loop_step",
                    "branch_id": branch.branch_id,
                    "iteration": ctx.iteration,
                    "observation": step.observation,
                    "metadata": step.metadata,
                },
                outputs={
                    "observation": step.observation,
                    "score": step.score,
                    "next_count": len(step.next_states),
                    "stop": step.stop,
                    "model_outputs": step.model_outputs or {},
                },
                parent_id=branch.last_step_id,
                duration=step.duration,
                cost=step.cost,
            )
            branch.last_step_id = step_id
            branch.steps += 1
            total_steps += 1
            branch.score = step.score

            if step.score > (best.score if best else float("-inf")):
                best = branch
                best_step_result = step

            if step.stop or step.score >= self.target_score:
                branch.recorder.record_done(
                    summary=f"Stopping branch '{branch.branch_id}' at score={step.score}",
                    parent_id=branch.last_step_id,
                )
                continue

            if not step.next_states:
                branch.recorder.record_done(
                    summary=f"No next state proposed from branch '{branch.branch_id}'",
                    parent_id=branch.last_step_id,
                )
                continue

            # First candidate keeps current branch alive; remaining candidates
            # become forked branches from this same step.
            sorted_next = sorted(
                step.next_states,
                key=lambda item: float(item.get("score", step.score)),
                reverse=True,
            )
            candidates = sorted_next[: self.branch_width]

            primary_applied = False
            for idx, cand in enumerate(candidates):
                if "state" not in cand:
                    continue
                if not primary_applied:
                    branch.state = dict(cand["state"])
                    if "best" in cand:
                        branch.best_state = dict(cand["state"])
                    queue.append(branch)
                    primary_applied = True
                    continue

                # Additional candidates are first forked and then queued separately.
                forked_recorder = branch.recorder.fork(
                    from_step_id=branch.last_step_id,
                    branch_id=branch_factory.branch_id("fork"),
                )
                forked_branch = LoopBranch(
                    branch_id=branch_factory.branch_id("alt"),
                    state=dict(cand["state"]),
                    recorder=forked_recorder,
                    last_step_id=branch.last_step_id,
                    score=float(cand.get("score", step.score)),
                    best_state=dict(cand["state"]),
                )
                forked_branch.recorder.record_think(
                    {
                        "event": "branch_from_parent",
                        "parent_branch": branch.branch_id,
                        "origin_step": branch.last_step_id,
                        "candidate_index": idx,
                        "candidate_score": cand.get("score", step.score),
                    },
                    parent_id=forked_branch.last_step_id,
                )
                queue.append(forked_branch)
                all_seen.append(forked_branch)

            if self.autosave_steps and branch.steps % self.autosave_steps == 0:
                branch.recorder.save(self.runs_dir / f"{branch.recorder.run.run_id}-{branch.steps}.tine")

            if len(queue) > self.max_branches:
                queue = deque(sorted(queue, key=lambda item: item.score, reverse=True)[: self.max_branches])

        if best is None:
            raise RuntimeError("loop engine did not execute any step")

        artifacts: list[Path] = []
        for item in all_seen:
            artifacts.append(item.recorder.save(self.runs_dir / f"{item.recorder.run.run_id}.tine"))
            if item is best:
                item.recorder.record_done(
                    summary="Selecting best branch for final artifact",
                    parent_id=item.last_step_id,
                )
                artifacts.append(item.recorder.save(self.runs_dir / f"{item.recorder.run.run_id}.best.tine"))

        return LoopExecutionResult(
            best=best,
            all_branches=all_seen,
            best_step_result=best_step_result,
            artifacts=artifacts,
        )
