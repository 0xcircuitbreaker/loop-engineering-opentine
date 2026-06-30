"""A deterministic numeric-threshold loop strategy.

The strategy repeatedly proposes candidates closer to a target number and lets the
engine branch between improvements.
"""

from __future__ import annotations

from loopforge.engine import LoopStepContext, LoopStepResult


def numeric_refinement(
    *,
    target: float = 42.0,
    step_size: float = 3.5,
    tol: float = 0.01,
):
    """Return a step function targeting a numeric optimum.

    This is useful as a loop-engine smoke test without external providers.
    """

    def _step(context: LoopStepContext) -> LoopStepResult:
        current = float(context.current_state.get("x", 0.0))
        delta = target - current
        score = -abs(delta)
        observation = f"branch {context.branch_id} at x={current:.4f} (delta={delta:.4f})"

        # stopping
        if abs(delta) <= tol:
            return LoopStepResult(
                observation=observation,
                next_states=[],
                score=score,
                stop=True,
                metadata={"final": True, "x": current, "target": target},
            )

        # deterministic candidate stepping in both directions
        candidates = [
            {"state": {"x": current + (step_size if delta > 0 else -step_size), "step": context.iteration}, "score": -abs(target - (current + (step_size if delta > 0 else -step_size)))},
            {"state": {"x": current + (delta / 2), "step": context.iteration}, "score": -abs(target - (current + (delta / 2)))},
        ]

        # avoid duplicates
        unique = []
        seen = {current}
        for c in candidates:
            xv = c["state"]["x"]
            if xv not in seen:
                seen.add(xv)
                unique.append(c)

        return LoopStepResult(
            observation=observation,
            next_states=unique,
            score=score,
            model_outputs={"x": current, "target": target, "delta": delta},
            metadata={"strategy": "numeric_refinement", "branch": context.branch_id},
        )

    return _step

