# opentine-loop-engineering

A public, reusable setup for **Loop Engineering** backed by [`opentine`](https://github.com/0xcircuitbreaker/opentine): every loop step is a recorded node in a content-addressed `.tine` graph, with branch/fork/replay semantics for complex iterative systems.

## What this repo gives you

- `LoopEngine` for stateful, branchable, iterative loops.
- `LoopRecorder` wrapper around opentine `Run` for reliable provenance capture.
- Built-in deterministic strategies (`numeric_refinement`, `text_rewrite`) for local testing.
- CLI entrypoint for running loops and verifying artifacts.
- A testing scaffold to keep your loop contracts honest.
- Artifact-first workflow: every branch ends with a `.tine` file.

## Core design

```text
goal/state -> LoopEngine step_fn -> LoopStepResult
             -> recorder.model(step)
             -> branch fan-out -> forked branch runs via opentine
             -> completed runs -> tine artifacts
```

Each loop iteration gets saved as an opentine step node, so:

- You can **fork from any step** and re-run alternatives.
- You can **compare branches** with `Run.diff(...)`.
- You can **verify integrity** before replay or audit.
- You can build Nth-degree recursion without losing history.

## Quick start

```bash
# Install local editable
cd /path/to/opentine-loop-engineering
pip install -e .

# Run numeric refinement loop
loopforge run-numeric 42 --start 0 --max-steps 25

# Run text rewrite loop
loopforge run-text "clear and concise status update" --start "start draft" --good-word clear

# Verify run
loopforge verify ./<run-id>.tine

# Compare two branches
loopforge compare left.tine right.tine
```

## Repository structure

- `src/loopforge/engine.py` — generic loop orchestration and branching logic.
- `src/loopforge/recorder.py` — opentine integration.
- `src/loopforge/strategies/` — deterministic strategy examples.
- `src/loopforge/cli.py` — executable interface.
- `tests/` — unit tests for engine + recorder behavior.
- `.github/workflows/ci.yml` — CI gate.

## Example: branchable numeric loop

```python
from loopforge import LoopEngine
from loopforge.strategies import numeric_refinement

engine = LoopEngine(
    step_fn=numeric_refinement(target=42),
    max_steps=30,
    max_branches=4,
    branch_width=2,
)

result = engine.run(
    goal="Find x close to 42",
    initial_state={"x": 0.0},
    context="local optimization smoke test",
)

print(result.best.branch_id)
print(result.best.state)
print(result.best.score)
```

Every active branch and each final candidate emits a `.tine` file under `~/.local/share/loopforge`.

## Why this helps with loop engineering

- **Complex loops**: branch fan-out and continuation at each iteration.
- **Controlled divergence**: cap active branches and branch width.
- **Auditability**: every step has parent links, timestamped costs, and integrity checksum.
- **Restartability**: fork and resume from any step boundary.

## API highlights

- `LoopEngine`:
  - `run(goal: str, initial_state: dict[str, Any], context: str = "") -> LoopExecutionResult`
- `LoopRecorder`:
  - `record_model`, `record_done`, `record_error`, `fork`, `save`
- `LoopStepResult`:
  - `observation`, `next_states`, `score`, `stop`, `metadata`

## Contributing

Run:

```bash
pytest -q
ruff check src tests
```


## Notes

. All generated `.tine` artifacts are written under `~/.local/share/loopforge` by default.



## Policy and model-backed execution

```python
from loopforge import LoopEngine, LoopPolicy
from loopforge.models import StaticModelAdapter, build_json_model_step
from loopforge.engine import LoopStepContext

policy = LoopPolicy(
    max_steps=12,
    max_total_cost=0.75,
    min_score=0.95,
    max_duration_seconds=30.0,
)

adapter = StaticModelAdapter(
    text='{"observation":"ok","score":0.97,"stop":true,"next_states":[]}',
    cost=0.01,
)

# Parse JSON responses shaped as {observation, score, stop, next_states, metadata}
def prompt_fn(ctx: LoopStepContext) -> str:
    return f"[{ctx.branch_id}] improve text: {ctx.current_state.get('text')}"

step_fn = build_json_model_step(adapter=adapter, prompt_fn=prompt_fn)
engine = LoopEngine(step_fn=step_fn, policy=policy, max_steps=5)
result = engine.run(goal="policy-aware loop", initial_state={"text": "start"})
print(result.best.score, result.best.recorder.run.run_id)
```

## MCP integration

A built-in MCP server exposes loop artifacts as tools for editor/agent automation:

- `list_run_artifacts`
- `show_run_tool`
- `diff_run_artifacts`
- `fork_run_artifact`

Start it with:

```bash
loopforge-mcp --runs-dir ~/.local/share/loopforge
```

