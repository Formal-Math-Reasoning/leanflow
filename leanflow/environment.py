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

import subprocess
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from filelock import FileLock
from .utils import logger, get_available_cpus
from .errors import LeanEnvironmentError, LeanBuildError, LeanTimeoutError

# Repository URLs
GIT_REPL = "https://github.com/leanprover-community/repl.git"
GIT_MATHLIB = "https://github.com/leanprover-community/mathlib4.git"

# Timeout constants (in seconds)
GIT_CLONE_TIMEOUT = 1800
LAKE_CACHE_TIMEOUT = 900
LAKE_BUILD_TIMEOUT = None

class EnvironmentManager:
    """Manages the download, build, and storage of Lean environments in a persistent user data location (~/.leanflow)."""
    def __init__(self, repl_config: Dict[str, Any]):
        """Initialize the EnvironmentManager.

        Args:
            repl_config (dict[str, Any]): Configuration dictionary containing paths or version info.
        """
        self.config = repl_config
        self.base_path = Path(self.config.get("base_path", Path.home() / ".leanflow"))
        self.environments_dir = self.base_path / "envs"
        self.environments_dir.mkdir(parents=True, exist_ok=True)

        # Configurable timeouts (override defaults via config)
        self.git_clone_timeout = self.config.get("git_clone_timeout", GIT_CLONE_TIMEOUT)
        self.lake_cache_timeout = self.config.get("lake_cache_timeout", LAKE_CACHE_TIMEOUT)
        self.lake_build_timeout = self.config.get("lake_build_timeout", LAKE_BUILD_TIMEOUT)

        self.cpus = get_available_cpus()

        logger.debug(f"Using environment base directory: {self.environments_dir}")

    def resolve_config(self) -> dict[str, str]:
        """Resolves the user config into a final dictionary with project_path and repl_path.

        This method handles various configuration scenarios:
        - If `project_path` is specified, it assumes valid paths.
        - If `lean_version` is specified, it ensures the environment exists or builds it.

        Returns:
            (dict[str, str]): A dictionary containing 'project_path' and 'repl_path'.

        Raises:
            LeanEnvironmentError: If a specified project path does not exist,
                or if the config is invalid (missing both project_path and lean_version).
        """

        # if project path specified, return project and repl paths (no build)
        if self.config.get("project_path") is not None:
            project_path = Path(self.config["project_path"])
            if not project_path.is_dir():
                raise LeanEnvironmentError(f"The specified 'project_path' does not exist: '{project_path}'")
            
            repl_exe_name = "repl.exe" if sys.platform == "win32" else "repl"
            repl_path = self.config.get("repl_path") or str(project_path / ".lake" / "build" / "bin" / repl_exe_name)
            
            return {"project_path": str(project_path), "repl_path": repl_path}

        # no project path -> check for existing environment or build new
        if self.config.get("lean_version") is not None:
            project_path, repl_path = self._get_or_build_env()
            return {"project_path": str(project_path), "repl_path": str(repl_path)}

        raise LeanEnvironmentError("Invalid 'repl_config'. Must contain either 'project_path' or 'lean_version'.")

    def _normalize_version(self, version: str) -> str:
        """Prepends 'v' to version numbers if they are plain (e.g., 4.21.0 -> v4.21.0).

        Args:
            version (str): The version string.

        Returns:
            (str): The normalized version string.
        """
        if version and version[0].isdigit():
            return f"v{version}"
        return version

    def _get_or_build_env(self) -> tuple[Path, Path]:
        """Ensures the specified environment exists, building it if necessary.

        Returns:
            (tuple[Path, Path]): A tuple containing (project_path, repl_executable_path).
        """

        lean_version = self._normalize_version(self.config["lean_version"])
        require_mathlib = self.config.get("require_mathlib", True)
        
        env_dir_name = f"lean-{lean_version}" + ("_mathlib" if require_mathlib else "")
        env_path = self.environments_dir / env_dir_name
        
        repl_path = env_path / "repl"
        repl_exe_name = "repl.exe" if sys.platform == "win32" else "repl"
        repl_executable_path = repl_path / ".lake" / "build" / "bin" / repl_exe_name
        
        mathlib_path = env_path / "mathlib" if require_mathlib else None
        project_path = mathlib_path if require_mathlib else repl_path

        is_fully_built = repl_executable_path.is_file() and (not require_mathlib or mathlib_path.is_dir())

        if is_fully_built:
            logger.debug(f"Found existing, valid environment in: '{env_path}'")
            return project_path, repl_executable_path

        # If not fully built, acquire a lock to build it.
        lock_path = self.environments_dir / f"{env_dir_name}.lock"
        logger.debug(f"Environment not found or incomplete. Acquiring lock to build at '{env_path}'")
        with FileLock(str(lock_path)):
            is_fully_built_in_lock = repl_executable_path.is_file() and (not require_mathlib or mathlib_path.is_dir())
            if is_fully_built_in_lock:
                logger.debug("Environment was built by another process. Using existing build.")
                return project_path, repl_executable_path
            
            self._build_environment(env_path, repl_path, mathlib_path, lean_version, require_mathlib)
        
        return project_path, repl_executable_path
    
    def shallow_clone(
        self,
        repo_url: str,
        target_path: Path,
        branch: str,
        timeout: int = GIT_CLONE_TIMEOUT,
    ):
        """Clone a git repository with a single branch.

        Args:
            repo_url: URL of the git repository.
            target_path: Local path to clone into.
            branch: Branch or tag to clone.
            timeout: Timeout in seconds.

        Raises:
            LeanBuildError: If git clone fails.
            LeanTimeoutError: If git clone times out.
        """
        cmd = ["git", "clone", "--single-branch", "--branch", branch, repo_url, str(target_path)]

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise LeanBuildError(
                    f"Git clone failed for {repo_url}",
                    command=cmd,
                    return_code=result.returncode,
                )
            logger.debug(f"Successfully cloned {repo_url} to {target_path}")

        except subprocess.TimeoutExpired:
            # Clean up partial clone
            if target_path.exists():
                shutil.rmtree(target_path)
            raise LeanTimeoutError(
                f"Git clone timed out after {timeout} seconds",
                timeout_seconds=timeout,
                operation=f"git clone {repo_url}",
            )

    def _run_lake_command(
        self,
        args: list,
        cwd: Path,
        timeout: Optional[int] = LAKE_BUILD_TIMEOUT,
        description: str = "lake command",
    ) -> subprocess.CompletedProcess:
        """Run a lake command with timeout and output capture.

        Args:
            args: Command arguments (e.g., ["lake", "build"]).
            cwd: Working directory for the command.
            timeout: Timeout in seconds.
            description: Human-readable description for error messages.

        Returns:
            CompletedProcess with captured stdout/stderr.

        Raises:
            LeanBuildError: If the command fails.
            LeanTimeoutError: If the command times out.
        """
        cmd = args if isinstance(args, list) else [args]

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise LeanBuildError(
                    f"{description} failed",
                    command=cmd,
                    return_code=result.returncode,
                )
            return result

        except subprocess.TimeoutExpired:
            raise LeanTimeoutError(
                f"{description} timed out after {timeout} seconds",
                timeout_seconds=timeout,
                operation=" ".join(cmd),
            )

    def _build_environment(
        self,
        env_path: Path,
        repl_path: Path,
        mathlib_path: Optional[Path],
        lean_version: str,
        require_mathlib: bool,
    ):
        """Clones repositories to a specific version and runs lake build.

        Args:
            env_path: Path to the environment directory.
            repl_path: Path to the REPL directory.
            mathlib_path: Path to the Mathlib directory (None if not required).
            lean_version: The Lean version tag.
            require_mathlib: Whether Mathlib is required.

        Raises:
            LeanBuildError: If a build command fails.
            LeanTimeoutError: If a build command times out.
            LeanEnvironmentError: Unexpected error.
        """
        logger.info(f"Building new environment at: '{env_path}'. This may take a few minutes...")

        # Clean up partial builds if they exist
        if env_path.exists():
            shutil.rmtree(env_path)
        env_path.mkdir(parents=True, exist_ok=True)

        try:
            # Always clone the REPL
            if not repl_path.exists():
                logger.debug(f"Cloning REPL branch '{lean_version}' to '{repl_path}'")
                self.shallow_clone(GIT_REPL, repl_path, lean_version)

            if require_mathlib and mathlib_path:
                # Clone Mathlib if required
                if not mathlib_path.exists():
                    logger.debug(f"Cloning Mathlib branch '{lean_version}' to '{mathlib_path}'")
                    self.shallow_clone(GIT_MATHLIB, mathlib_path, lean_version)

                    # Try to get cached build artifacts
                    try:
                        logger.debug("Attempting to fetch Mathlib cache...")
                        self._run_lake_command(
                            ["lake", "exe", "cache", "get"],
                            cwd=mathlib_path,
                            timeout=self.lake_cache_timeout,
                            description="lake exe cache get",
                        )
                        logger.debug("Mathlib cache retrieved successfully")
                    except (LeanBuildError, LeanTimeoutError) as e:
                        logger.warning(
                            f"Command 'lake exe cache get' failed: {e}. "
                            "Falling back to manual build (this will be slow)..."
                        )

                # Build Mathlib first
                logger.debug(f"Running 'lake build' in {mathlib_path}...")
                self._run_lake_command(
                    ["lake", "build"],
                    cwd=mathlib_path,
                    description="Mathlib build",
                )

            # Always build the REPL
            logger.debug(f"Running 'lake build' in {repl_path}...")
            self._run_lake_command(
                ["lake", "build"],
                cwd=repl_path,
                description="REPL build",
            )

            logger.success(f"Environment built successfully at '{env_path}'")

        except (LeanBuildError, LeanTimeoutError):
            # Re-raise our custom errors as-is
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during environment build: {e}")
            raise LeanEnvironmentError(f"Unexpected error during environment build: {e}")

    def _get_command_version(self, command: str) -> Optional[str]:
        """Get version string for a command, or None if not available."""
        try:
            result = subprocess.run(
                [command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            raise LeanEnvironmentError(f"Failed to get version string for '{command}'.")
        return None

    def diagnose(self) -> dict[str, Any]:
        """Run diagnostics for troubleshooting environment issues.

        Returns a dictionary with system information useful for debugging
        environment setup problems, especially on HPC/SLURM clusters.

        Returns:
            Dict containing diagnostic information.
        """
        diagnostics = {
            "python_version": sys.version,
            "platform": sys.platform,
            "home_directory": str(Path.home()),
            "base_path": str(self.base_path),
            "environments_dir": str(self.environments_dir),
            "base_path_exists": self.base_path.exists(),
            "base_path_writable": os.access(self.base_path, os.W_OK) if self.base_path.exists() else False,
            "cpu_count": self.cpus,
            "lake_version": self._get_command_version("lake"),
            "git_version": self._get_command_version("git"),
            "elan_version": self._get_command_version("elan"),
            "lean_env_vars": {
                k: v for k, v in os.environ.items()
                if any(x in k.upper() for x in ["LEAN", "LAKE", "ELAN", "MATHLIB"])
            },
            "path_env": os.environ.get("PATH", ""),
        }

        # Check for existing environments
        if self.environments_dir.exists():
            diagnostics["existing_environments"] = [
                d.name for d in self.environments_dir.iterdir() if d.is_dir()
            ]
        else:
            diagnostics["existing_environments"] = []

        return diagnostics

    def diagnose_pretty(self) -> str:
        """Run diagnostics and return a formatted string for display.

        Returns:
            Formatted diagnostic information as a string.
        """
        diag = self.diagnose()
        lines = [
            "=== LeanFlow Environment Diagnostics ===",
            "",
            f"Python: {diag['python_version']}",
            f"Platform: {diag['platform']}",
            f"CPU count: {diag['cpu_count']}",
            "",
            "--- Paths ---",
            f"Home: {diag['home_directory']}",
            f"Base path: {diag['base_path']} (exists: {diag['base_path_exists']}, writable: {diag['base_path_writable']})",
            f"Environments: {diag['environments_dir']}",
            "",
            "--- Tools ---",
            f"Lake: {diag['lake_version'] or 'NOT FOUND'}",
            f"Git: {diag['git_version'] or 'NOT FOUND'}",
            f"Elan: {diag['elan_version'] or 'NOT FOUND'}",
            "",
            "--- Existing Environments ---",
        ]

        if diag["existing_environments"]:
            for env in diag["existing_environments"]:
                lines.append(f"  - {env}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("--- Relevant Environment Variables ---")
        if diag["lean_env_vars"]:
            for k, v in diag["lean_env_vars"].items():
                lines.append(f"  {k}={v}")
        else:
            lines.append("  (none set)")

        return "\n".join(lines)