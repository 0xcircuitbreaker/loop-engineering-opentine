from __future__ import annotations

from pathlib import Path

from opentine import Run

from loopforge import LoopEngine, LoopPolicy
from loopforge.engine import LoopStepContext, LoopStepResult
from loopforge.models import StaticModelAdapter, build_json_model_step
from loopforge import mcp_server


def test_loop_policy_embedding_and_cost_cap() -> None:
    def step_fn(ctx: LoopStepContext) -> LoopStepResult:
        return LoopStepResult(
            observation=f"iter={ctx.iteration}",
            next_states=[{"state": {"value": ctx.current_state.get("value", 0) + 1}}],
            score=ctx.iteration,
            cost=0.25,
        )

    policy = LoopPolicy(max_total_cost=0.2)
    engine = LoopEngine(step_fn=step_fn, max_steps=5, policy=policy, runs_dir="/tmp/loopforge-tests")
    result = engine.run(goal="cost cap", initial_state={"value": 0})

    artifact = result.best.recorder.run
    assert artifact.status.value == "completed"
    assert result.best.score == 1

    loaded = Run.load(result.artifacts[0])
    assert loaded.policies["loop"]["max_total_cost"] == 0.2


def test_static_json_model_step_parser() -> None:
    payload = (
        '{"observation":"ok","score":0.5,"stop":false,"next_states":'
        '[{"state":{"x":1},"score":0.9}],"metadata":{"tag":"x"}}'
    )
    adapter = StaticModelAdapter(text=payload, cost=0.1, duration=0.7)

    def prompt_fn(ctx: LoopStepContext) -> str:
        return f"goal={ctx.iteration}"

    step_fn = build_json_model_step(adapter=adapter, prompt_fn=prompt_fn)
    result = step_fn(LoopStepContext(iteration=1, branch_id="root", current_state={"x": 0}, best_score=0.0, history_len=0, run_id="", elapsed_seconds=0.0, accumulated_cost=0.0))

    assert result.observation == "ok"
    assert result.next_states == [{"state": {"x": 1}, "score": 0.9}]
    assert result.score == 0.5
    assert result.duration == 0.7
    assert result.cost == 0.1
    assert result.model_outputs["adapter_meta"]["kind"] == "static"


def test_static_json_model_step_fallback_on_invalid_json() -> None:
    adapter = StaticModelAdapter(text="not json")

    def prompt_fn(ctx: LoopStepContext) -> str:
        return "go"

    step_fn = build_json_model_step(adapter=adapter, prompt_fn=prompt_fn)
    result = step_fn(LoopStepContext(iteration=2, branch_id="root", current_state={}, best_score=None, history_len=0, run_id="", elapsed_seconds=0.0, accumulated_cost=0.0))

    assert result.stop is True
    assert result.score == float("-inf")
    assert "Model parse failed" in result.observation


def test_mcp_tools_over_local_runs(tmp_path: Path) -> None:
    def step_fn(ctx: LoopStepContext) -> LoopStepResult:
        return LoopStepResult(
            observation="step",
            next_states=[],
            score=0.0,
        )

    engine = LoopEngine(
        step_fn=step_fn,
        max_steps=1,
        runs_dir=str(tmp_path),
    )
    result = engine.run(goal="mcp", initial_state={"x": 1})

    listing = mcp_server.list_runs(runs_dir=str(tmp_path))
    assert listing
    assert len(listing) >= 1

    run = result.best.recorder.run
    shown = mcp_server.show_run(run.run_id, runs_dir=str(tmp_path))
    assert shown["run_id"] == run.run_id
    assert shown["top_steps"]
    assert shown["step_count_total"] == shown["step_count"]

    # Fork from the last step and verify the new run loads.
    source = run.run_id
    src_path = f"{run.run_id}.tine"
    loaded = Run.load(tmp_path / src_path)
    forked = mcp_server.fork_run(source, from_step=loaded.graph.order[0], runs_dir=str(tmp_path))

    assert forked["run_id"]
    assert Path(forked["path"]).exists()

    diffed = mcp_server.diff_runs(source, forked["run_id"], runs_dir=str(tmp_path))
    assert "common_ancestor" in diffed
