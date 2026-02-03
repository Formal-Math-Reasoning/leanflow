import asyncio
from typing import Union, Optional

from .client import Client
from ..utils import Environment, LeanError

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
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def close(self):
        """Close the client connection and clean up resources."""
        loop = self._get_loop()
        loop.run_until_complete(self._client.close())
        if not loop.is_closed():
            loop.close()
        self._loop = None

    def run(self, command: str, env: Optional[Union[Environment, int]] = None) -> Union[Environment, LeanError]:
        """Run a Lean command synchronously via the server.

        Args:
            command (str): A single command string.
            env (Optional[int]): Optional environment ID to run in.

        Returns:
            Environment or LeanError.
        """
        return self._get_loop().run_until_complete(self._client.run(command, env))
    
    def run_list(self, commands: list[str], env: Optional[Union[Environment, int]] = None) -> list[Union[Environment, LeanError]]:
        """Run Lean commands synchronously via the server.

        Args:
            commands (list[str]): A list of command strings.
            env (Optional[int]): Optional environment ID to run in.

        Returns:
            List of results (Environment or LeanError).
        """
        return self._get_loop().run_until_complete(self._client.run_list(commands, env))

    def status(self) -> bool:
        """Check server status synchronously.

        Returns:
            (bool): True if the server is running and accessible, False otherwise.
        """
        return self._get_loop().run_until_complete(self._client.status())

    def __enter__(self) -> "SyncClient":
        """Enter context manager."""
        self._get_loop().run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        loop = self._get_loop()
        loop.run_until_complete(self._client.__aexit__(exc_type, exc_val, exc_tb))
        if not loop.is_closed():
            loop.close()
        self._loop = None