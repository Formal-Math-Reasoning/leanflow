from importlib.metadata import version
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

__version__ = version("leanflow")

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