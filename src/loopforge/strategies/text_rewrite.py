"""A deterministic text refinement strategy.

Each step edits the prompt with a simple expansion/concision policy and scores by
length and keyword coverage. Good for demonstrating branchable "content quality"
loops.
"""

from __future__ import annotations

from loopforge.engine import LoopStepContext, LoopStepResult

def text_rewrite(goal: str = "", good_word: str = "clear", max_len: int = 40):
    """Return a deterministic text-loop step function."""
    goal = goal.strip()

    def _step(context: LoopStepContext) -> LoopStepResult:
        txt = context.current_state.get("text", "")
        if not txt:
            txt = goal or "".join(context.run_id.split("-")[:3])

        # score based on presence of target word and length target
        score = 0.0
        if goal and goal in txt:
            score += 0.5
        if good_word in txt:
            score += 0.5
        score -= abs(len(txt) - max_len) / 100.0

        observation = f"version={context.iteration} text_len={len(txt)} score={score:.3f}"

        if context.iteration >= 8 or score >= 0.95:
            return LoopStepResult(
                observation=observation,
                next_states=[],
                score=score,
                stop=True,
                metadata={"text": txt, "quality": score, "goal": goal},
            )

        next_states = []
        # Candidate A: expand with goal
        expanded = (txt + " "+goal).strip() if goal else f"{txt} {good_word}"
        # Candidate B: compress/rephrase
        compact = " ".join(sorted(set((txt.replace(",", "").split()), key=str.lower))[:max_len])

        next_states.append({"state": {"text": expanded}, "score": score + 0.2})
        next_states.append({"state": {"text": compact}, "score": score + 0.1})

        return LoopStepResult(
            observation=observation,
            next_states=next_states,
            score=score,
            model_outputs={"text": txt},
            metadata={"strategy": "text_rewrite"},
        )

    return _step

