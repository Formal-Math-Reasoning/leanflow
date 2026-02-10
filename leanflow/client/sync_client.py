from typing import Union, Optional

from .client import Client
from ..utils import Environment, LeanError, _AsyncRunner

class SyncClient:
    """Synchronous wrapper around the Client class. Provides a blocking API for interacting with a LeanFlow server without needing to use async/await.

    Example:
        ```python
        from leanflow import SyncClient

        with SyncClient(base_url="http://localhost:8000") as client:
            result = client.run("#eval 1 + 1")
            print(result)
        ```
    """

    def __init__(self, base_url: str, timeout: int = 300):
        """Initialize the SyncClient with the given configuration.

        Args:
            base_url (str): URL of the LeanFlow server.
            timeout (int): Request timeout in seconds.
        """
        self._client = Client(base_url, timeout)
        self._runner = _AsyncRunner()

    def close(self):
        """Close the client connection and clean up resources."""
        self._runner.run(self._client.close())
        self._runner.close()

    def run(self, command: str, env: Optional[Union[Environment, int]] = None) -> Union[Environment, LeanError]:
        """Run a Lean command synchronously via the server.

        Args:
            command (str): A single command string.
            env (Optional[int]): Optional environment ID to run in.

        Returns:
            Environment or LeanError.
        """
        return self._runner.run(self._client.run(command, env))

    def run_list(self, commands: list[str], env: Optional[Union[Environment, int]] = None) -> list[Union[Environment, LeanError]]:
        """Run Lean commands synchronously via the server.

        Args:
            commands (list[str]): A list of command strings.
            env (Optional[int]): Optional environment ID to run in.

        Returns:
            List of results (Environment or LeanError).
        """
        return self._runner.run(self._client.run_list(commands, env))

    def status(self) -> bool:
        """Check server status synchronously.

        Returns:
            (bool): True if the server is running and accessible, False otherwise.
        """
        return self._runner.run(self._client.status())

    def __enter__(self) -> "SyncClient":
        """Enter context manager."""
        self._runner.run(self._client.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._runner.run(self._client.__aexit__(exc_type, exc_val, exc_tb))
        self._runner.close()
