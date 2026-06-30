"""Loop engineering toolkit built on opentine artifacts.

Core idea: treat every loop iteration as a first-class graph node and make
branching, retries, and audits explicit, replayable, and verifiable.
"""

from .engine import LoopEngine, LoopExecutionResult, LoopFactory, LoopStepContext, LoopStepResult
from .recorder import LoopRecorder

__all__ = [
    "LoopEngine",
    "LoopExecutionResult",
    "LoopFactory",
    "LoopStepContext",
    "LoopStepResult",
    "LoopRecorder",
]

__version__ = "0.1.0"
