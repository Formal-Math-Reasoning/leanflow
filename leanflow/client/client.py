# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import asyncio
from pathlib import Path
from typing import Union, Optional, Any

import httpx
import dacite

from ..utils import setup_logger, logger, get_env_id, Environment, LeanError, Pos
from .utils import _parse_pos_from_string

class Client:
    """An asynchronous client for interacting with a remote LeanFlow Server.

    Example:
        ```python
        import asyncio
        from leanflow import Client

        async def main():
            async with Client("http://localhost:8000") as client:
                result = await client.run("#eval 1 + 1")
                print(result)

        asyncio.run(main())
        ```
    """
    def __init__(
        self,
        base_url: str,
        timeout: Optional[int] = None,
        max_connections: int = 1000,
        max_keepalive_connections: int = 100,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: Optional[str] = None,
    ):
        """Initializes the Client.

        Args:
            base_url (str): The base URL of the leanflow-server.
            timeout (Optional[int]): Request timeout in seconds.
            max_connections (int): Maximum number of connections.
            max_keepalive_connections (int): Maximum number of keepalive connections.
            log_dir (Optional[Union[str, Path]]): Explicit directory to save log files.
            log_level (Optional[str]): The logging level (e.g., "INFO", "DEBUG").
        """
        setup_logger(log_dir, log_level)
        logger.info
        self.base_url = base_url
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )

        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=limits,
        )

        self.dacite_config = dacite.Config(
            type_hooks={
                Pos: _parse_pos_from_string,
                Optional[Pos]: _parse_pos_from_string
            },
            check_types=False
        )

    async def _deserialize(self, data: dict[str, Any]) -> Union[Environment, LeanError]:
        """Deserializes a dictionary from the server into an Environment or LeanError object.

        Args:
            data (dict[str, Any]): The raw data dictionary from the server.

        Returns:
            (Union[Environment, LeanError]): The parsed Environment or LeanError object.
        """
        try:
            if "message" in data:
                return dacite.from_dict(LeanError, data, config=self.dacite_config)
            else:
                return dacite.from_dict(Environment, data, config=self.dacite_config)
        except dacite.DaciteError as e:
            logger.error(f"Client failed to parse server response: {e}\nRaw data: {data}")
            return LeanError(message=f"Client failed to parse server response: {e}", source="dacite")

    async def run(self, command: str, env: Optional[Union[Environment, int]] = None) -> Union[Environment, LeanError]:
        """Executes a single Lean command.

        Args:
            command (str): The Lean command to execute.
            env (Optional[Union[Environment, int]]): The environment ID to run in.

        Returns:
            (Union[Environment, LeanError]): The result of the command.
        """
        payload = {"command": command}
        env_id = get_env_id(env)
        if env_id is not None:
            payload["env"] = env_id

        try:
            response = await self._client.post("/run", json=payload)
            response.raise_for_status()
            data = response.json()
            result = await self._deserialize(data["result"])
            if isinstance(result, LeanError):
                logger.error(result.message)
            return result
        except httpx.HTTPStatusError as e:
            msg = f"Server error: {e.response.status_code}. Response: {e.response.text}"
            logger.error(msg)
            return LeanError(message=msg, source="server")
        except httpx.TimeoutException as e:
            msg = f"Request {e.request} timed out."
            logger.error(msg)
            return LeanError(message=msg, source="timeout")
        except Exception as e:
            msg = f"An unexpected client-side error occurred: {e}"
            msg += f"\nMake sure that the server is running at {self.base_url}."
            logger.error(msg)
            return LeanError(message=msg, source="server")

    async def run_list(self, commands: list[str], env: Optional[Union[Environment, int]] = None) -> list[Union[Environment, LeanError]]:
        """Executes a list of commands sequentially, in the same environment and stopping on the first error.

        Args:
            commands (list[str]): A list of commands.
            env (Optional[Union[Environment, int]]): The environment to run the commands in.

        Returns:
            List of results.
        """
        results = []

        for cmd in commands:
            res = await self.run(cmd, env)
            results.append(res)
            
            if isinstance(res, Environment):
                # set the environment for the next command
                env = res
            else:
                logger.error(f"Command failed: {res}")
                break
        return results

    async def status(self) -> bool:
        """Checks if the remote server is running and accessible.

        Returns:
            (bool): True if the server returns status 'ok', False otherwise.
        """
        try:
            response = await self._client.get("/status")
            response.raise_for_status()
            return response.json().get("status") == "ok"
        except Exception:
            return False

    async def __aenter__(self):
        return self

    async def close(self):
        """Closes the underlying HTTP client."""
        await self._client.aclose()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()