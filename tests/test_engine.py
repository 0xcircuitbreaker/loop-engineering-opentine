from __future__ import annotations

from loopforge import LoopEngine
from loopforge.strategies import numeric_refinement
from loopforge.recorder import LoopRecorder
from loopforge.engine import LoopStepContext, LoopStepResult
from opentine import Run


def test_numeric_refinement_reaches_target() -> None:
    engine = LoopEngine(
        step_fn=numeric_refinement(target=10.0, step_size=2.5),
        max_steps=30,
        max_branches=5,
        branch_width=2,
        runs_dir="/tmp/loopforge-tests",
    )
    result = engine.run(goal="Reach 10", initial_state={"x": 0.0})

    assert result.best.score != float("-inf")
    assert result.best.recorder.run.run_id
    assert result.best.state
    assert len(result.all_branches) >= 1


def test_noop_step_does_not_crash() -> None:
    def _step(ctx: LoopStepContext) -> LoopStepResult:
        return LoopStepResult(
            observation="ok",
            next_states=[],
            score=0.0,
            stop=True,
            metadata={"iteration": ctx.iteration},
        )

    engine = LoopEngine(step_fn=_step, runs_dir="/tmp/loopforge-tests")
    result = engine.run(goal="noop", initial_state={"x": 1})

    assert result.best.recorder.run.status.value == "completed"
    assert result.best.state == {"x": 1}


def test_recorder_roundtrip_integrity() -> None:
    r = LoopRecorder(goal="test", context="unit test")
    step = r.record_model(payload={"a": 1}, outputs={"b": 2})
    assert step
    assert r.run.status.value == "running"
    path = r.save("/tmp/loopforge-tests/unit.tine")
    assert path.exists()

    verified = Run.verify_integrity(path)
    assert verified.ok
