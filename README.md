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
from loopforge import LoopEngine, numeric_refinement

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
