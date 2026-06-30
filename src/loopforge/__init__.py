"""Loop engineering toolkit built on opentine artifacts.

Core idea: treat every loop iteration as a first-class graph node and make
branching, retries, and audits explicit, replayable, and verifiable.
"""

from .engine import LoopEngine, LoopExecutionResult, LoopFactory, LoopStepContext, LoopStepResult
from .policy import LoopPolicy, LoopPolicyViolation
from .recorder import LoopRecorder
from .models import (
    LoopModelAdapter,
    LoopModelOutput,
    OpenAICompatAdapter,
    StaticModelAdapter,
    build_json_model_step,
)
from .mcp_server import (
    DEFAULT_RUNS_DIR,
    build_server,
    diff_runs,
    fork_run,
    list_runs,
    show_run,
)

__all__ = [
    "LoopEngine",
    "LoopExecutionResult",
    "LoopFactory",
    "LoopStepContext",
    "LoopStepResult",
    "LoopRecorder",
    "LoopPolicy",
    "LoopPolicyViolation",
    "LoopModelAdapter",
    "LoopModelOutput",
    "OpenAICompatAdapter",
    "StaticModelAdapter",
    "build_json_model_step",
    "DEFAULT_RUNS_DIR",
    "build_server",
    "diff_runs",
    "fork_run",
    "list_runs",
    "show_run",
]

__version__ = "0.2.1"
