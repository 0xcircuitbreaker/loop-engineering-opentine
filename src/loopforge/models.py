"""Model adapters and loop-oriented strategy factories."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import request, error

from .engine import LoopStepContext, LoopStepResult


class LoopModelOutput:
    """Output from a model adapter."""

    def __init__(
        self,
        text: str,
        *,
        cost: float = 0.0,
        duration: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.cost = cost
        self.duration = duration
        self.meta = meta or {}


class LoopModelAdapter(Protocol):
    """Minimal interface for model providers."""

    def complete(self, prompt: str) -> LoopModelOutput: ...


@dataclass
class StaticModelAdapter:
    """Deterministic adapter for tests and demos.

    Returns the configured `text` for each request.
    """

    text: str = ""
    cost: float = 0.0
    duration: float = 0.0

    def complete(self, prompt: str) -> LoopModelOutput:
        return LoopModelOutput(
            self.text,
            cost=self.cost,
            duration=self.duration,
            meta={"prompt_len": len(prompt), "kind": "static"},
        )


@dataclass
class OpenAICompatAdapter:
    """Small generic OpenAI-compatible chat-completion adapter.

    Useful when you want to plug in `gpt`, `groq`, `together`, `deepseek`, and
    other OpenAI protocol providers without adding extra dependencies.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 20.0
    temperature: float = 0.2
    max_tokens: int | None = None
    cost_per_million_input: float | None = None
    cost_per_million_output: float | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAI-compatible adapter needs an API key. Set api_key=... or OPENAI_API_KEY."
            )

    def complete(self, prompt: str) -> LoopModelOutput:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        req = request.Request(
            url=self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8")
                data = json.loads(raw)
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc

        elapsed = time.perf_counter() - started

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No completion choices returned")

        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        in_tokens = float(usage.get("prompt_tokens", 0) or 0)
        out_tokens = float(usage.get("completion_tokens", 0) or 0)
        if self.cost_per_million_input is not None and self.cost_per_million_output is not None:
            cost = (in_tokens / 1_000_000 * self.cost_per_million_input) + (
                out_tokens / 1_000_000 * self.cost_per_million_output
            )
        else:
            cost = 0.0

        return LoopModelOutput(
            text,
            cost=cost,
            duration=elapsed,
            meta={"usage": usage, "provider": "openai-compat"},
        )


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Attempt to parse JSON from raw model text.

    Supports raw JSON or fenced JSON blocks.
    """

    candidates = [text.strip()]
    if "```" in text:
        chunks = text.split("```")
        candidates = [c.strip() for c in chunks if c.strip() and not c.strip().startswith("json")]
        if candidates:
            candidates = [
                c[4:].strip() if c.lower().startswith("json") else c
                for c in candidates
            ] + [text.strip()]

    for raw in candidates:
        try:
            return json.loads(raw)
        except Exception:
            continue
    raise ValueError("Model output was not parseable JSON")


def build_json_model_step(
    *,
    adapter: LoopModelAdapter,
    prompt_fn: Callable[[LoopStepContext], str],
    fallback_step: Callable[[LoopStepContext, Exception], LoopStepResult] | None = None,
) -> Callable[[LoopStepContext], LoopStepResult]:
    """Build a LoopStepResult-returning step function from a JSON model adapter.

    The model output is expected to contain keys:
    - observation: str
    - score: float
    - stop: bool
    - next_states: list[{"state": {...}, "score": optional float}]
    - metadata: optional dict
    """

    def _fallback(ctx: LoopStepContext, exc: Exception) -> LoopStepResult:
        if fallback_step is not None:
            return fallback_step(ctx, exc)
        return LoopStepResult(
            observation=f"Model parse failed at {ctx.iteration}: {exc}",
            next_states=[],
            score=float("-inf"),
            stop=True,
            metadata={"error": str(exc), "iteration": ctx.iteration},
        )

    def _step(context: LoopStepContext) -> LoopStepResult:
        prompt = prompt_fn(context)
        try:
            output = adapter.complete(prompt)
            data = _extract_json_from_text(output.text)
            next_states = []
            raw_next = data.get("next_states") or []
            for candidate in raw_next:
                if not isinstance(candidate, dict):
                    continue
                state = candidate.get("state")
                if not isinstance(state, dict):
                    continue
                score = candidate.get("score")
                next_states.append(
                    {
                        "state": state,
                        "score": float(score)
                        if isinstance(score, int | float)
                        else context.best_score or 0.0,
                    }
                )

            return LoopStepResult(
                observation=str(data.get("observation", "")),
                next_states=next_states,
                score=float(data.get("score", context.best_score or 0.0)),
                stop=bool(data.get("stop", False)),
                model_outputs={
                    "text": output.text,
                    "raw_text": output.text,
                    "adapter_meta": output.meta,
                },
                duration=output.duration,
                cost=float(output.cost),
                metadata=data.get("metadata", {}),
            )
        except Exception as exc:  # pragma: no cover - parser/network boundary
            return _fallback(context, exc)

    return _step

