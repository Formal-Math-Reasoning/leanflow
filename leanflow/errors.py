from typing import Optional

class LeanFlowError(Exception):
    """Base exception for all LeanFlow errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.details is not None:
            return f"{self.message}\n\nDetails:\n{self.details}"
        return self.message


class LeanEnvironmentError(LeanFlowError):
    """Error during Lean environment setup or configuration.

    Raised when:
    - Environment directory cannot be created
    - Required paths are missing or inaccessible
    - Configuration is invalid
    """
    pass


class LeanBuildError(LeanFlowError):
    """Error during Lean/Lake build process.

    Raised when:
    - Git clone fails
    - Lake build fails
    - Lake cache retrieval fails
    """

    def __init__(
        self,
        message: str,
        command: Optional[list[str]] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        return_code: Optional[int] = None,
    ):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code

        details_parts = []
        if command is not None:
            details_parts.append(f"Command: {' '.join(command)}")
        if return_code is not None:
            details_parts.append(f"Return code: {return_code}")
        if stdout is not None:
            details_parts.append(f"Stdout:\n{stdout}")
        if stderr is not None:
            details_parts.append(f"Stderr:\n{stderr}")

        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


class LeanTimeoutError(LeanFlowError):
    """Error when an operation times out.

    Raised when:
    - Git clone times out
    - Lake build times out
    - REPL command times out
    """

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        self.timeout = timeout
        self.operation = operation

        details_parts = []
        if operation is not None:
            details_parts.append(f"Operation: {operation}")
        if timeout is not None:
            details_parts.append(f"Timeout: {timeout} seconds")

        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


class LeanConnectionError(LeanFlowError):
    """Error in REPL communication.

    Raised when:
    - REPL process unexpectedly closes
    - Communication protocol fails
    - JSON parsing fails
    """
    pass


class LeanMemoryError(LeanFlowError):
    """Error in REPL memory.

    Raised when:
    - Max number of memory restarts exceeded
    """
    pass


class LeanValueError(LeanFlowError):
    """Error in REPL type arguments.

    Raised when:
    - Invalid or missing arguments
    """
    pass


class LeanHeaderError(LeanFlowError):
    """Error during REPL header execution.

    Raised when:
    - Header command fails to execute
    - Header produces an error from Lean
    """

    def __init__(
        self,
        message: str,
        header: Optional[str] = None,
        lean_error: Optional[str] = None,
    ):
        self.header = header
        self.lean_error = lean_error

        details_parts = []
        if header is not None:
            # Truncate long headers for readability
            header_preview = header[:500] + "..." if len(header) > 500 else header
            details_parts.append(f"Header:\n{header_preview}")
        if lean_error is not None:
            details_parts.append(f"Lean error:\n{lean_error}")

        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


class LeanServerError(LeanFlowError):
    """Error in server operations.

    Raised when:
    - Server fails to start
    - Worker pool exhausted
    - Server-side execution fails
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        self.status_code = status_code
        self.response_body = response_body

        details_parts = []
        if status_code is not None:
            details_parts.append(f"Status code: {status_code}")
        if response_body is not None:
            details_parts.append(f"Response:\n{response_body}")

        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)
