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

from pathlib import Path
from typing import Union, Optional

from .repl import REPL
from ..utils import Environment, LeanError, ProofState, _AsyncRunner

class SyncREPL:
    """Synchronous wrapper around REPL for convenience.

    Provides a blocking API for interacting with the Lean REPL without
    needing to use async/await.

    Example:
        ```python
        from leanflow import SyncREPL

        with SyncREPL(lean_version="v4.24.0") as repl:
            result = repl.run("theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry")
            print(result)
        ```
    """

    def __init__(
        self,
        lean_version: Optional[str] = None,
        repl_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        **kwargs,
    ):
        """Initialize the SyncREPL with the given configuration.

        Args:
            lean_version (Optional[str]): The Lean version to use (e.g. "4.24.0").
            repl_path (Optional[Path]): Path to the Lean REPL executable.
            project_path (Optional[Path]): Path to a local Lean project.
            **kwargs (Any): Other arguments passed to REPL
                (timeout, max_memory_mb).
        """
        self._repl = REPL(lean_version, repl_path, project_path, **kwargs)
        self._runner = _AsyncRunner()

    def start(self):
        """Start the REPL process.

        This is called automatically when using the context manager.
        """
        self._runner.run(self._repl._ensure_running_async())

    def close(self):
        """Close the REPL process and clean up resources.

        This is called automatically when using the context manager.
        """
        self._runner.run(self._repl.close_async())
        self._runner.close()

    def run(self, command: str, env: Optional[Union[Environment, int]] = None) -> Union[Environment, LeanError]:
        """Run a Lean command synchronously.

        Args:
            command: A single command string.
            env: Optional environment to run in.

        Returns:
            Environment or LeanError.
        """
        return self._runner.run(self._repl.run(command, env))

    def run_list(self, commands: list[str], env: Optional[Union[Environment, int]] = None) -> list[Union[Environment, LeanError]]:
        """Run Lean commands synchronously.

        Args:
            commands: A list of command strings.
            env: Optional environment to run in.

        Returns:
            List of results (Environment or LeanError).
        """
        return self._runner.run(self._repl.run_list(commands, env))

    def run_file(self, path: Union[str, Path], return_all_states: bool = False) -> Union[Environment, LeanError]:
        """Run a Lean file synchronously.

        Args:
            path: Path to the Lean file.
            return_all_states: Whether to return all intermediate states.

        Returns:
            Environment or LeanError.
        """
        return self._runner.run(
            self._repl.run_file_async(Path(path), return_all_states)
        )

    def run_tactic(self, tactic: str, state: ProofState) -> Union[ProofState, LeanError]:
        """Run a tactic in a proof state synchronously.

        Args:
            tactic: The tactic to run.
            state: The current proof state.

        Returns:
            New ProofState or LeanError.
        """
        return self._runner.run(
            self._repl.run_tactic_async(tactic, state)
        )

    def __enter__(self) -> "SyncREPL":
        """Enter context manager."""
        self._runner.run(self._repl.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._runner.run(self._repl.__aexit__(exc_type, exc_val, exc_tb))
        self._runner.close()
