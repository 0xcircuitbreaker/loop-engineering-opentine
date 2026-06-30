"""MCP server for loopforge artifacts.

Exposes lightweight tools to list, inspect, compare, and fork loop artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from opentine import Run


try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - optional dependency behavior
    FastMCP = None  # type: ignore[assignment]

DEFAULT_RUNS_DIR = Path("~/.local/share/loopforge").expanduser()


def _normalize_ref(ref: str, runs_dir: Path) -> Path:
    p = Path(ref)
    if p.suffix == ".tine" and p.exists():
        return p

    # bare filename without suffix
    candidate = runs_dir / f"{ref}.tine"
    if candidate.exists():
        return candidate

    # prefix match on run_id
    for file in runs_dir.glob("*.tine"):
        try:
            run = Run.load(file)
        except Exception:
            continue
        if run.run_id == ref or run.run_id.startswith(ref):
            return file
    raise FileNotFoundError(f"Could not resolve artifact for '{ref}'")


def _run_metadata(run: Run) -> dict[str, Any]:
    cost = run.cost_breakdown()
    return {
        "run_id": run.run_id,
        "status": run.status.value if run.status is not None else None,
        "step_count": len(run.graph.order),
        "created_at": run.created_at,
        "manifest": run.manifest,
        "cost": {
            "total_cost": cost.total_cost,
            "total_tokens": cost.total_tokens,
            "by_model": cost.by_model,
            "by_kind": cost.by_kind,
        },
        "policies": run.policies,
        "tags": run.tags,
        "last_ref": run.refs.get("main", ""),
    }


def list_runs(runs_dir: str = str(DEFAULT_RUNS_DIR)) -> list[dict[str, Any]]:
    """List run artifacts available in a directory."""
    root = Path(runs_dir).expanduser()
    out: list[dict[str, Any]] = []
    for file in sorted(root.glob("*.tine")):
        try:
            run = Run.load(file)
        except Exception:
            continue
        data = _run_metadata(run)
        data["path"] = str(file)
        out.append(data)
    return out


def show_run(ref: str, runs_dir: str = str(DEFAULT_RUNS_DIR)) -> dict[str, Any]:
    """Load an artifact and return a compact summary.

    Includes the top-level manifest and first 10 steps.
    """
    path = _normalize_ref(ref, Path(runs_dir).expanduser())
    run = Run.load(path)
    meta = _run_metadata(run)
    top_steps = [run.graph.steps[step_id] for step_id in run.graph.order[:10]]
    meta["top_steps"] = [
        {
            "id": step.id,
            "kind": step.kind.value,
            "inputs": step.inputs,
            "outputs": step.outputs,
            "duration": step.duration,
            "cost": step.cost,
        }
        for step in top_steps
    ]
    meta["step_count_total"] = len(run.graph.order)
    return meta


def diff_runs(left: str, right: str, runs_dir: str = str(DEFAULT_RUNS_DIR)) -> dict[str, Any]:
    """Return a machine-friendly diff summary for two artifacts."""
    root = Path(runs_dir).expanduser()
    run_a = Run.load(_normalize_ref(left, root))
    run_b = Run.load(_normalize_ref(right, root))
    d = run_a.diff(run_b)
    return {
        "common_ancestor": d.common_ancestor,
        "only_a": [s.id for s in d.only_a],
        "only_b": [s.id for s in d.only_b],
        "changed": [
            {
                "step_a": getattr(c.step_a, 'id', None),
                "step_b": getattr(c.step_b, 'id', None),
                "fields": [
                    {"field": delta.field, "before": delta.before, "after": delta.after, "path": delta.path}
                    for delta in getattr(c, "fields", [])
                ],
            }
            for c in d.changed
        ],
    }


def fork_run(
    ref: str,
    from_step: str,
    branch: str = "operator-fork",
    runs_dir: str = str(DEFAULT_RUNS_DIR),
) -> dict[str, str]:
    """Fork an existing artifact from a given step.

    Returns the new artifact path and id.
    """
    root = Path(runs_dir).expanduser()
    source_path = _normalize_ref(ref, root)
    run = Run.load(source_path)
    forked = run.fork(from_step_id=from_step, branch=branch)
    new_path = root / f"{forked.run_id}.tine"
    forked.save(new_path)
    return {"run_id": forked.run_id, "path": str(new_path)}


def build_server(runs_dir: str = str(DEFAULT_RUNS_DIR)) -> "FastMCP":
    """Build an MCP server instance.

    Raises if mcp dependency is unavailable.
    """
    if FastMCP is None:
        raise RuntimeError("mcp dependency is not installed. Install with `pip install loopforge[mcp]`")

    app = FastMCP("loopforge")

    @app.tool()
    def list_run_artifacts() -> list[dict[str, Any]]:  # type: ignore[override]
        return list_runs(runs_dir=runs_dir)

    @app.tool()
    def show_run_tool(run: str) -> dict[str, Any]:  # type: ignore[override]
        return show_run(run, runs_dir=runs_dir)

    @app.tool()
    def diff_run_artifacts(left: str, right: str) -> dict[str, Any]:  # type: ignore[override]
        return diff_runs(left, right, runs_dir=runs_dir)

    @app.tool()
    def fork_run_artifact(run: str, from_step: str, branch: str = "mcp-fork") -> dict[str, str]:  # type: ignore[override]
        return fork_run(run, from_step=from_step, branch=branch, runs_dir=runs_dir)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run loopforge MCP server")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = parser.parse_args()
    server = build_server(runs_dir=args.runs_dir)
    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
