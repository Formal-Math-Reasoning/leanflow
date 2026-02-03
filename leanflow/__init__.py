"""LeanFlow - A Python Interface for Lean 4.

LeanFlow provides both synchronous and asynchronous interfaces for interacting
with the Lean 4 theorem prover, supporting both local REPL and remote server modes.

Async API (recommended for performance):
    - REPL: Local Lean REPL interaction
    - Client: Remote server interaction

Sync API (simpler, blocking):
    - SyncREPL: Synchronous wrapper around REPL
    - SyncClient: Synchronous wrapper around Client
"""

from .repl import REPL, SyncREPL
from .client import Client, SyncClient
from .server import Server
from .utils import LeanError, Environment
from .environment import EnvironmentManager
from .metrics import (
    Metric,
    BatchMetric,
    LLMAsAJudge,
    TypeCheck,
    BEqPlus,
    BEqL,
    BEq,
    EquivRfl,
    ConJudge,
    LLMGrader
)
from .errors import (
    LeanFlowError,
    LeanEnvironmentError,
    LeanBuildError,
    LeanTimeoutError,
    LeanConnectionError,
    LeanMemoryError,
    LeanValueError,
    LeanHeaderError,
    LeanServerError,
)

__version__ = "0.0.2"

__all__ = [
    # Async API
    "REPL",
    "Client",
    # Sync API
    "SyncREPL",
    "SyncClient",
    # Server
    "Server",
    # Utilities
    "LeanError",
    "Environment",
    "EnvironmentManager",
    # Metrics
    "Metric",
    "BatchMetric",
    "LLMAsAJudge",
    "TypeCheck",
    "BEqPlus",
    "BEqL",
    "BEq",
    "EquivRfl",
    "ConJudge",
    "LLMGrader",
    # Error types
    "LeanFlowError",
    "LeanEnvironmentError",
    "LeanBuildError",
    "LeanTimeoutError",
    "LeanConnectionError",
    "LeanMemoryError",
    "LeanValueError",
    "LeanHeaderError",
    "LeanServerError",
    # Version
    "__version__",
]