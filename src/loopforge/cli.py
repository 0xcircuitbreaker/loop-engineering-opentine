"""CLI for running loop-engine experiments and summarizing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from opentine import Run

from .engine import LoopEngine
from .strategies import numeric_refinement, text_rewrite

app = typer.Typer(help="Run branchable loops with opentine provenance traces")


def _write_json(payload: dict, out: Optional[Path]) -> None:
    text = json.dumps(payload, indent=2)
    if out is None:
        print(text)
    else:
        Path(out).write_text(text)
        print(f"Wrote {out}")


@app.command()
def run_numeric(
    target: float = typer.Argument(42.0, help="Numeric target to converge on"),
    start: float = typer.Option(0.0, help="Starting x value"),
    max_steps: int = typer.Option(20, help="Loop budget"),
    branches: int = typer.Option(4, help="Maximum simultaneous branches"),
    branch_width: int = typer.Option(2, help="Fan-out at each split"),
    out: Optional[Path] = typer.Option(None, help="Optional JSON summary path"),
) -> None:
    """Run the built-in numeric-refinement loop."""
    engine = LoopEngine(
        step_fn=numeric_refinement(target=target),
        max_steps=max_steps,
        max_branches=branches,
        branch_width=branch_width,
    )
    result = engine.run(goal=f"Find x near {target}", initial_state={"x": start})

    payload = {
        "goal": f"Find x near {target}",
        "start": start,
        "best_run_id": result.best.recorder.run.run_id,
        "best_score": result.best.score,
        "best_state": result.best.state,
        "artifact_count": len(result.artifacts),
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
    out: Optional[Path] = typer.Option(None, help="Optional JSON summary path"),
) -> None:
    """Run the built-in text-refinement loop."""
    engine = LoopEngine(step_fn=text_rewrite(goal=goal, good_word=good_word), max_steps=25, max_branches=3, branch_width=2)
    result = engine.run(goal=f"Rewrite toward: {goal}", initial_state={"text": start})

    payload = {
        "goal": goal,
        "start": start,
        "best_run_id": result.best.recorder.run.run_id,
        "best_score": result.best.score,
        "best_text": result.best.state,
        "artifact_count": len(result.artifacts),
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


if __name__ == "__main__":  # pragma: no cover
    app()
