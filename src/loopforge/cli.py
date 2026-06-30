"""CLI for running loop-engine experiments and managing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from opentine import Run

from .engine import LoopEngine
from .models import StaticModelAdapter, build_json_model_step
from .policy import LoopPolicy
from .strategies import numeric_refinement, text_rewrite
from . import mcp_server

try:  # pragma: no cover - optional runtime branch
    import typer
except Exception:  # pragma: no cover - dependency optional in core docs
    typer = None

if typer is not None:
    app = typer.Typer(help="Run branchable loops with opentine provenance traces")
else:  # pragma: no cover
    class _NoOp:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("typer is required for CLI usage")

    app = _NoOp()


def _write_json(payload: dict[str, Any], out: Optional[Path]) -> None:
    text = json.dumps(payload, indent=2)
    if out is None:
        print(text)
    else:
        Path(out).write_text(text)
        print(f"Wrote {out}")


def _policy_from_opts(
    max_cost: float | None,
    max_duration: float | None,
    min_score: float | None,
    max_steps: int,
) -> LoopPolicy:
    return LoopPolicy(
        max_steps=max_steps,
        max_total_cost=max_cost,
        max_duration_seconds=max_duration,
        min_score=min_score,
    )


def _run_common(
    goal: str,
    step_fn,
    *,
    max_steps: int,
    branches: int,
    branch_width: int,
    max_cost: float | None,
    max_duration: float | None,
    min_score: float | None,
    out: Optional[Path],
    summary_label: str,
) -> dict[str, Any]:
    engine = LoopEngine(
        step_fn=step_fn,
        max_steps=max_steps,
        max_branches=branches,
        branch_width=branch_width,
        policy=_policy_from_opts(max_cost, max_duration, min_score, max_steps),
    )
    result = engine.run(goal=goal, initial_state={"value": 0})

    payload = {
        "goal": goal,
        "best_run_id": result.best.recorder.run.run_id,
        "best_score": result.best.score,
        "best_state": result.best.state,
        "summary": summary_label,
        "artifact_count": len(result.artifacts),
        "artifacts": [str(path) for path in result.artifacts],
    }
    _write_json(payload, out)
    return payload


if typer is not None:
    @app.command()
    def run_numeric(
        target: float = typer.Argument(42.0, help="Numeric target to converge on"),
        start: float = typer.Option(0.0, help="Starting x value"),
        max_steps: int = typer.Option(20, help="Loop budget"),
        branches: int = typer.Option(4, help="Maximum simultaneous branches"),
        branch_width: int = typer.Option(2, help="Fan-out at each split"),
        max_cost: float | None = typer.Option(None, help="Optional total cost budget"),
        max_duration: float | None = typer.Option(
            None,
            help="Optional wall-clock budget in seconds",
        ),
        min_score: float | None = typer.Option(None, help="Stop a branch if score threshold is met"),
        out: Optional[Path] = typer.Option(None, help="Optional JSON summary path"),
    ) -> None:
        """Run the built-in numeric-refinement loop."""
        engine = LoopEngine(
            step_fn=numeric_refinement(target=target),
            max_steps=max_steps,
            max_branches=branches,
            branch_width=branch_width,
            policy=_policy_from_opts(max_cost, max_duration, min_score, max_steps),
        )
        result = engine.run(goal=f"Find x near {target}", initial_state={"x": start})

        payload = {
            "goal": f"Find x near {target}",
            "start": start,
            "best_run_id": result.best.recorder.run.run_id,
            "best_score": result.best.score,
            "best_state": result.best.state,
            "artifact_count": len(result.artifacts),
            "artifacts": [str(p) for p in result.artifacts],
        }
        _write_json(payload, out)
        print(f"✅ best branch: {result.best.branch_id} score={result.best.score}")
        print(f"   run: {result.best.recorder.run.run_id}")
        print(f"   state: {result.best.state}")
        if result.best_step_result:
            print(f"   best-step observation: {result.best_step_result.observation}")

    @app.command()
    def run_text(
        goal: str = typer.Argument(..., help="Target quality phrase"),
        start: str = typer.Option("", help="Initial text seed"),
        good_word: str = typer.Option("clear", help="Quality keyword"),
        max_steps: int = typer.Option(25, help="Loop budget"),
        branches: int = typer.Option(3, help="Maximum simultaneous branches"),
        branch_width: int = typer.Option(2, help="Fan-out at each split"),
        max_cost: float | None = typer.Option(None, help="Optional total cost budget"),
        max_duration: float | None = typer.Option(None, help="Optional wall-clock budget in seconds"),
        min_score: float | None = typer.Option(None, help="Stop a branch if score threshold is met"),
        out: Optional[Path] = typer.Option(None, help="Optional JSON summary path"),
    ) -> None:
        """Run the built-in text-refinement loop."""
        engine = LoopEngine(
            step_fn=text_rewrite(goal=goal, good_word=good_word),
            max_steps=max_steps,
            max_branches=branches,
            branch_width=branch_width,
            policy=_policy_from_opts(max_cost, max_duration, min_score, max_steps),
        )
        result = engine.run(goal=f"Rewrite toward: {goal}", initial_state={"text": start})

        payload = {
            "goal": goal,
            "start": start,
            "best_run_id": result.best.recorder.run.run_id,
            "best_score": result.best.score,
            "best_state": result.best.state,
            "artifact_count": len(result.artifacts),
            "artifacts": [str(p) for p in result.artifacts],
        }
        _write_json(payload, out)

    @app.command()
    def run_model_json_demo(
        goal: str = typer.Argument(..., help="Target objective"),
        static_response: str = typer.Argument(
            '{"observation":"generated","score":0.5,"stop":true,"next_states":[]}',
            help="Static JSON returned by the fake adapter",
        ),
        max_steps: int = typer.Option(3, help="Loop budget"),
        branches: int = typer.Option(2, help="Maximum simultaneous branches"),
        branch_width: int = typer.Option(1, help="Fan-out at each split"),
        out: Optional[Path] = typer.Option(None, help="Optional JSON summary path"),
    ) -> None:
        """Run a JSON adapter-backed strategy for model-provider integrations."""

        adapter = StaticModelAdapter(text=static_response)

        def prompt_fn(ctx):
            return f"[{ctx.run_id}] Goal:{goal} state:{ctx.current_state}"

        step_fn = build_json_model_step(adapter=adapter, prompt_fn=prompt_fn)
        engine = LoopEngine(
            step_fn=step_fn,
            max_steps=max_steps,
            max_branches=branches,
            branch_width=branch_width,
        )
        result = engine.run(goal=f"Model-backed: {goal}", initial_state={"seed": ""})

        payload = {
            "goal": goal,
            "best_run_id": result.best.recorder.run.run_id,
            "best_score": result.best.score,
            "best_state": result.best.state,
            "artifacts": [str(p) for p in result.artifacts],
        }
        _write_json(payload, out)

    @app.command()
    def verify(artifact: Path = typer.Argument(..., help="Path to .tine artifact")):
        """Verify a .tine artifact from disk."""
        result = Run.verify_integrity(artifact)
        print(f"verified={result.ok}, digest={result.digest}")

    @app.command()
    def compare(
        left: Path = typer.Argument(..., help="Base run artifact"),
        right: Path = typer.Argument(..., help="Competing run artifact"),
    ) -> None:
        """Print opentine diff summary for two run artifacts."""
        run_a = Run.load(left)
        run_b = Run.load(right)
        d = run_a.diff(run_b)

        print(f"common_ancestor={d.common_ancestor}")
        print(f"only_in_a={len(d.only_a)}")
        print(f"only_in_b={len(d.only_b)}")
        print(f"changed={len(d.changed)}")

    @app.command()
    def mcp_server(  # type: ignore[no-redef]
        runs_dir: str = typer.Option(str(mcp_server.DEFAULT_RUNS_DIR), help="Run directory"),
    ) -> None:
        """Start loopforge MCP server for IDE/tooling integration."""
        app = mcp_server.build_server(runs_dir=runs_dir)
        app.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    if typer is None:
        raise RuntimeError("Install loopforge with [dev] dependencies to use the CLI")
    app()
