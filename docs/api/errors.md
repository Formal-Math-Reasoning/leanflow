# Error Types

LeanFlow provides structured error types for robust error handling. All errors inherit from the base `LeanFlowError`.


## Quick Reference

| Error | When It's Raised | Key Attributes |
|-------|------------------|----------------|
| `LeanBuildError` | Compilation of Lean or Lake dependencies fails. | `message`, `stderr`, `return_code` |
| `LeanTimeoutError` | An operation exceeds the configured timeout limit. | `message`, `timeout`, `operation` |
| `LeanHeaderError` | The startup header script fails to execute. | `message`, `header`, `lean_error` |
| `LeanConnectionError` | The Client connection to the server fails. | `message` |
| `LeanMemoryError` | The REPL exceeds configured memory limits. | `message` |
| `LeanValueError` | Invalid arguments are passed to a function. | `message` |
| `LeanEnvironmentError` | Errors raised when creating or locating the Lean environment. | `message` |
| `LeanServerError` | Issues raised by the LeanFlow server. | `message` |

## Handling Errors

```python
from leanflow import SyncREPL, LeanBuildError, LeanTimeoutError, LeanEnvironmentError, LeanValueError

try:
    with SyncREPL(lean_version="4.24.0") as repl:
        result = repl.run("#check 1 + 1")
        print(result)
except LeanBuildError as e:
    print(f"Build failed: {e.message}")
except LeanTimeoutError as e:
    print(f"Timed out: {e.operation}")
except LeanEnvironmentError as e:
    print(f"Environment error: {e.message}")
except LeanValueError as e:
    print(f"Invalid argument: {e.message}")
```

### Suppressing Header Errors

By default, header failures raise `LeanHeaderError`. To log warnings instead:

```python
with SyncREPL(
    lean_version="4.24.0",
    header="import Mathlib",
    fail_on_header_error=False  # Logs warning instead
) as repl:
    ...
```

## Diagnosing Environment Issues

For setup problems (especially on HPC clusters):

```python
from leanflow import EnvironmentManager

manager = EnvironmentManager({"lean_version": "4.24.0"})
print(manager.diagnose_pretty())
```
