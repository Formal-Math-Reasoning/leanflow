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
import json
import sys
from asyncio.subprocess import Process, PIPE
from pathlib import Path
from typing import Union, Any, Optional

import dacite
import psutil
from omegaconf import OmegaConf

from ..utils import setup_logger, logger, get_env_id, LeanError, Environment, ProofState
from ..environment import EnvironmentManager
from ..errors import LeanEnvironmentError, LeanConnectionError, LeanMemoryError, LeanHeaderError


class REPL:
    """REPL is an asynchronous, robust wrapper for the Lean REPL (https://github.com/leanprover-community/repl).
    It automatically manages its memory and restarts itself.

    Run as an async context manager (`async with`) or to
    manually call `close_async` when done to ensure proper cleanup.

    Example:
        ```python
        import asyncio
        from leanflow import REPL

        async def main():
            async with REPL(lean_version="4.24.0") as repl:
                result = await repl.run("theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry")
                print(result)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        lean_version: Optional[str] = None,
        repl_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        timeout: int = 300,
        header: Optional[str] = None,
        dacite_config: Optional[dacite.Config] = None,
        fail_on_header_error: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: Optional[str] = None,
        debug: bool = False,
        manage_memory: bool = True,
        max_memory_mb: int = 8192,
        max_restart_attempts: int = 3,
        **kwargs
    ):
        """Initializes the REPL.

        Args:
            lean_version (Optional[str]): The Lean version to use (e.g. "4.21.0").
            repl_path (Optional[Path]): Path to the Lean REPL executable.
            project_path (Optional[Path]): Path to the Lean project directory.
            timeout (int): Timeout for the REPL process.
            header (Optional[str]): Header to be sent to the REPL.
            dacite_config (Optional[dacite.Config]): Configuration for dacite.
            fail_on_header_error (bool): If True, raises LeanHeaderError when header execution fails.
                If False, log the error and continue with base_env_id=None.
            log_dir (Optional[Union[str, Path]]): Explicit directory to save log files.
            log_level (Optional[str]): The logging level (e.g., "INFO", "DEBUG").
            debug (bool): If True, enables file logging even if log_dir is not provided.
            manage_memory (bool): If True, manages memory and restarts if needed.
            max_memory_mb (int): Maximum memory in MB before a restart is triggered.
            max_restart_attempts (int): Maximum consecutive restarts before raising MemoryError.
            **kwargs (Any): Additional keyword arguments passed to the EnvironmentManager.
        """
        setup_logger(log_dir, log_level, debug)

        # resolve enviroment configuration
        if lean_version is not None:
            kwargs["lean_version"] = lean_version
        if repl_path is not None:
            kwargs["repl_path"] = repl_path
        if project_path is not None:
            kwargs["project_path"] = project_path
        manager = EnvironmentManager(kwargs)
        resolved_config = manager.resolve_config()

        repl_path = resolved_config.get("repl_path")
        project_path = resolved_config.get("project_path")

        self.repl_path = Path(repl_path).absolute()
        self.project_path = Path(project_path).absolute()
        self.timeout = timeout
        self.header = header
        self.fail_on_header_error = fail_on_header_error
        self.dacite_config = dacite.Config(
            check_types=True, strict=True, strict_unions_match=True
        ) if dacite_config is None else dacite_config

        self._proc: Optional[Process] = None
        self._lock = asyncio.Lock()
        self.base_env_id: Optional[int] = None

        self.manage_memory = manage_memory
        if manage_memory:
            self.max_memory_mb = max_memory_mb
            self.max_restart_attempts = max_restart_attempts
            self._restart_counter = 0

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "REPL":
        """Loads a REPL instance from a YAML file.

        Args:
            yaml_path (Path): Path to the YAML file.

        Returns:
            (REPL): The loaded REPL instance.
        """
        conf = OmegaConf.load(yaml_path)
        dacite_config = conf.pop("dacite_config", None)
        if dacite_config:
            dacite_config = dacite.Config(**dacite_config)
        return cls(**conf, dacite_config=dacite_config)

    async def _start_interactive_async(self):
        """Starts the Lean REPL subprocess asynchronously.

        Raises:
            LeanEnvironmentError: If project path or REPL executable is not found, or other unexpected errors during startup.
            LeanHeaderError: Header execution error.
        """

        if not self.project_path or not self.project_path.is_dir():
            raise LeanEnvironmentError(f"The specified project_path '{self.project_path}' does not exist or is not a directory.")
        
        lakefile_lean_path = self.project_path / "lakefile.lean"
        lakefile_toml_path = self.project_path / "lakefile.toml"
        if not lakefile_lean_path.is_file() and not lakefile_toml_path.is_file():
            logger.warning(
                f"The project_path '{self.project_path}' does not contain a 'lakefile.lean' or 'lakefile.toml'. "
            )

        cwd = self.project_path

        if self.repl_path:
            if not self.repl_path.is_file():
                if not self.repl_path.is_dir():
                    raise LeanEnvironmentError(f"The specified repl_path '{self.repl_path}' does not exist or is not a directory.")
            
                repl_exe_name = "repl.exe" if sys.platform == "win32" else "repl"
                repl_executable = self.repl_path / ".lake" / "build" / "bin" / repl_exe_name
            
            else:
                repl_executable = self.repl_path
            
            if not repl_executable.is_file():
                raise LeanEnvironmentError(f"Could not find the REPL executable at the expected path: {repl_executable}. Please ensure the REPL project has been built correctly.")
            
            cmd_args = ["lake", "env", str(repl_executable)]
        else:
            cmd_args = ["lake", "exe", "repl"]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=cwd,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
            )
            logger.debug(f"Lean REPL process started with PID: {self._proc.pid}")
        except FileNotFoundError:
            logger.exception(f"Failed to start the Lean REPL. The command '{cmd_args[0]}' was not found. Please ensure 'lake' is installed and accessible in your system's PATH.")
            raise LeanEnvironmentError(f"Failed to start the Lean REPL. The command '{cmd_args[0]}' was not found. Please ensure 'lake' is installed and accessible in your system's PATH.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while starting the Lean REPL process: {e}")
            raise LeanEnvironmentError(f"An unexpected error occurred while starting the Lean REPL process: {e}")

        if self.header:
            # We directly submit the header command without acquiring the main lock
            header_req = json.dumps({"cmd": self.header}, ensure_ascii=False)
            output_json = await self._submit_request_async(header_req, startup=True)

            try:
                data = json.loads(output_json)
                if "env" in data:
                    self.base_env_id = data["env"]
                    logger.debug(f"Header executed successfully, base_env_id={self.base_env_id}")
                else:
                    error_msg = data.get("message", "Unknown error during header execution.")
                    if self.fail_on_header_error:
                        raise LeanHeaderError(
                            "Failed to initialize REPL with header",
                            header=self.header,
                            lean_error=error_msg,
                        )
                    else:
                        logger.warning(f"Header execution failed (continuing anyway): {error_msg}")
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON from header execution: {output_json}"
                if self.fail_on_header_error:
                    raise LeanHeaderError(
                        error_msg,
                        header=self.header,
                        lean_error=str(e),
                    )
                else:
                    logger.warning(error_msg)

    async def close_async(self):
        """Closes the Lean REPL subprocess and its entire process tree."""
        if not self._proc or self._proc.returncode is not None:
            return

        logger.debug(f"Closing Lean REPL process with PID: {self._proc.pid}")
        try:
            parent = psutil.Process(self._proc.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([parent] + children, timeout=3)
            for p in alive:
                p.kill()
        except psutil.NoSuchProcess:
            # Process already died
            pass
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")
        finally:
            self._proc = None

    async def _restart_async(self):
        """Restarts the Lean REPL subprocess asynchronously."""
        logger.warning(f"Restarting the Lean REPL with PID: {self._proc.pid}")
        await self.close_async()
        await self._start_interactive_async()

    async def _ensure_running_async(self):
        """Ensures the REPL process is running, starting it if necessary."""
        if not self._proc or self._proc.returncode is not None:
            await self._start_interactive_async()

    async def __aenter__(self) -> "REPL":
        """Enters the async context manager."""
        await self._ensure_running_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exits the async context manager."""
        await self.close_async()

    def __del__(self):
        """Garbage collection cleanup."""
        if hasattr(self, "_proc") and self._proc and self._proc.returncode is None:
            logger.warning(
                f"REPL object garbage collected but process {self._proc.pid} was still running. "
                "Use 'async with' or 'await repl.close_async()' for reliable cleanup."
            )
            self._proc.terminate()
    
    async def _submit_request_async(self, req: str, startup: bool = False) -> str:
        """Submits a request to the Lean REPL and awaits the full JSON response.

        Lock acquisition behavior:
        - Normal requests (startup=False): First ensures REPL is running, then acquires
          self._lock before I/O. The lock ensures sequential request/response pairs and
          prevents interleaved I/O, which would corrupt the protocol.
        - Startup requests (startup=True): Bypasses both the running check and uses the
          lock directly. This is used during initialization in _start_interactive_async
          when sending the header. The lock is still acquired to ensure sequential I/O.

        Args:
            req (str): The JSON-encoded request string.
            startup (bool): Whether this request is part of the startup sequence.
                If True, bypasses _ensure_running_async check (to avoid dealake dlock).

        Returns:
            (str): The JSON response string.

        Raises:
            LeanConnectionError: If the REPL process is not running.
        """
        if not startup:
            await self._ensure_running_async()
        
        async with self._lock:
            if not self._proc or self._proc.stdin is None or self._proc.stdout is None:
                raise LeanConnectionError("REPL process is not running.")

            try:
                self._proc.stdin.write(req.encode("utf-8") + b"\n\n")
                await self._proc.stdin.drain()

                # read chunked response
                buffer = bytearray()
                delimiter = b"\n\n"
                while True:
                    # read up to 4KB at a time
                    chunk = await asyncio.wait_for(self._proc.stdout.read(4096), self.timeout)
                    if not chunk:
                        raise LeanConnectionError("Lean REPL closed the connection (EOF).")
                    buffer.extend(chunk)
                    if delimiter in buffer:
                        break
                
                response_bytes, _, _ = buffer.partition(delimiter)
                response_str = response_bytes.decode("utf-8").strip()
                return response_str

            except asyncio.TimeoutError:
                logger.error("Request timed out. Restarting REPL.")
                await self._restart_async()
                return json.dumps({"message": "Request timed out.", "source": "timeout"})

            except LeanConnectionError as e:
                stderr_output = ""
                if self._proc and self._proc.stderr:
                    try:
                        stderr_output = (await asyncio.wait_for(self._proc.stderr.read(), timeout=1)).decode().strip()
                    except:
                        pass
                logger.error(f"Request failed: {type(e).__name__}. Restarting REPL.")
                if stderr_output:
                    logger.error(f"REPL stderr: {stderr_output}")
                await self._restart_async()
                logger.info("The REPL has been restarted.")
                # Return a JSON string that will be parsed into a LeanError
                return json.dumps({"message": f"Request failed ({type(e).__name__}).", "source": "connection"})

    def _get_memory_usage_mb(self) -> float:
        """Gets the total memory usage of the process tree in MB.

        Returns:
            (float): Memory usage in megabytes. Returns 0.0 if process is not running.
        """
        if not self._proc or self._proc.returncode is not None:
            return 0.0
        try:
            parent = psutil.Process(self._proc.pid)
            memory_bytes = parent.memory_info().rss
            for child in parent.children(recursive=True):
                memory_bytes += child.memory_info().rss
            return memory_bytes / (1024 * 1024)
        except psutil.NoSuchProcess:
            return 0.0

    async def _run_and_parse_async(self, request_payload: dict, data_class: type) -> Union[Any, LeanError]:
        """Helper to run a command and parse the result into a dataclass.

        Args:
            request_payload (dict): The dictionary to send as JSON.
            data_class (type): The dataclass type to parse the response into.

        Returns:
            (Union[Any, LeanError]): The parsed dataclass instance or a LeanError.

        Raises:
            MemoryError: If the maximum number of restart attempts is exceeded.
        """
        if self.manage_memory:
            current_memory = self._get_memory_usage_mb()
            if current_memory > self.max_memory_mb:
                logger.warning(f"Memory usage ({current_memory:.2f}MB) exceeds limit ({self.max_memory_mb}MB). Restarting REPL.")
                self._restart_counter += 1
                if self._restart_counter >= self.max_restart_attempts:
                    raise LeanMemoryError(f"Exceeded max restart attempts ({self.max_restart_attempts}) due to high memory usage.")
                
                await self._restart_async()
            else:
                self._restart_counter = 0

        try:
            request_json = json.dumps(request_payload, ensure_ascii=False)
        except Exception as e:
            msg = f"Failed to parse request as a JSON command: {e}"
            logger.error(msg)
            return LeanError(message=msg, source="JSON")
        output_json = await self._submit_request_async(request_json)


        try:
            data = json.loads(output_json)
        except json.JSONDecodeError:
            msg = f"Failed to decode JSON from REPL: {output_json}"
            logger.error(msg)
            return LeanError(message=msg, source="JSON")

        # Check if the response is a Lean error message
        if "message" in data and len(data) <= 2:
            return dacite.from_dict(data_class=LeanError, data=data, config=self.dacite_config)

        try:
            result = dacite.from_dict(data_class=data_class, data=data, config=self.dacite_config)
            # Post-process Environment to populate goals from messages
            if data_class == Environment and hasattr(result, "messages"):
                for msg in result.messages:
                    if "unsolved goals" in msg.data:
                        lines = msg.data.split("\n")
                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith("⊢"):
                                goal = stripped[1:].strip()
                                result.goals.append(goal)
                
                # Also populate goals from Sorries
                if hasattr(result, "sorries"):
                    for sorry in result.sorries:
                        if sorry.goal:
                            result.goals.append(sorry.goal)
            return result
        except dacite.DaciteError as e:
            msg = f"Failed to parse REPL response into {data_class.__name__}: {e}"
            logger.error(f"{msg}\nRaw data: {data}")
            return LeanError(message=msg, source="dacite")

    async def run_file_async(self, path: Path, return_all_states: bool = False) -> Union[Environment, LeanError]:
        """Runs a Lean file.

        Args:
            path (Path): Path to the Lean file.
            return_all_states (bool): Whether to return all intermediate states (not fully supported in return type yet).

        Returns:
            (Union[Environment, LeanError]): The resulting environment or an error.
        """
        payload = {"path": str(path), "allTactics": return_all_states}
        return await self._run_and_parse_async(payload, Environment)

    async def run_tactic_async(self, tactic: str, state: ProofState) -> Union[ProofState, LeanError]:
        """Runs a tactic in a given proof state.

        Args:
            tactic (str): The tactic to run.
            state (ProofState): The current proof state.

        Returns:
            (Union[ProofState, LeanError]): The new proof state or an error.
        """
        payload = {"tactic": tactic, "proofState": state.proofState}
        return await self._run_and_parse_async(payload, ProofState)

    async def pickle_env_async(self, path: Path, env: Environment) -> Union[Environment, LeanError]:
        """Pickles an environment to a file.

        Args:
            path (Path): Destination path.
            env (Environment): Environment to pickle.

        Returns:
            (Union[Environment, LeanError]): Result or error.
        """
        payload = {"pickleTo": str(path), "env": env.env}
        return await self._run_and_parse_async(payload, Environment)

    async def unpickle_env_async(self, path: Path) -> Union[Environment, LeanError]:
        """Unpickles an environment from a file.

        Args:
            path (Path): Source path.

        Returns:
            (Union[Environment, LeanError]): The unpickled environment or error.
        """
        payload = {"unpickleEnvFrom": str(path)}
        return await self._run_and_parse_async(payload, Environment)

    async def pickle_state_async(self, path: Path, state: ProofState) -> Union[ProofState, LeanError]:
        """Pickles a proof state to a file.

        Args:
            path (Path): Destination path.
            state (ProofState): Proof state to pickle.

        Returns:
            (Union[ProofState, LeanError]): Result or error.
        """
        payload = {"pickleTo": str(path), "proofState": state.proofState}
        return await self._run_and_parse_async(payload, ProofState)

    async def unpickle_state_async(self, path: Path) -> Union[ProofState, LeanError]:
        """Unpickles a proof state from a file.

        Args:
            path (Path): Source path.

        Returns:
            (Union[ProofState, LeanError]): The unpickled proof state or error.
        """
        payload = {"unpickleProofStateFrom": str(path)}
        return await self._run_and_parse_async(payload, ProofState)
    
    async def run(self, command: str, env: Optional[Union[Environment, int]] = None) -> Union[Environment, LeanError]:
        """Runs a Lean command in the specified environment.

        Args:
            command (str): The Lean command to execute.
            env (Optional[Union[Environment, int]]): The environment to run in.

        Returns:
            (Union[Environment, LeanError]): The resulting environment or an error.
        """
        payload = {"cmd": command}
        env_id = get_env_id(env)
        if env_id is None:
            env_id = self.base_env_id
        if env_id is not None:
            payload["env"] = env_id
        return await self._run_and_parse_async(payload, Environment)

    async def run_list(
        self, commands: list[str], env: Optional[Union[Environment, int]] = None
    ) -> list[Union[Environment, LeanError]]:
        """Runs a list of commands in the specified environment.

        Args:
            commands (list[str]): A list of command strings.
            env (Optional[Union[Environment, int]]): The Lean environment.
        
        Returns:
            List of results.
        """
        results = []

        for cmd in commands:
            res = await self.run(cmd, env)
            results.append(res)
            
            if isinstance(res, Environment):
                env = res
            else:
                logger.error("A command in the sequence failed.")
                break
        return results