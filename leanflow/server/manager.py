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
import itertools
import time
from typing import Dict, List, Optional, Union

from ..utils import setup_logger, logger, Environment, LeanError
from ..repl import REPL
from ..environment import EnvironmentManager

IDLE_TIMEOUT = 600
IDLE_CHECK_INTERVAL = 60

class StateManager:
    """Manages a pool of REPL workers using a queue-based leasing system."""
    def __init__(self, repl_pool: List[REPL], stateless: bool = False):
        """Initialize a StateManager with a pool of initialized REPLs.

        Args:
            repl_pool (List[REPL]): List of initialized REPL instances.
            stateless (bool): Whether to run in stateless mode.
        """
        max_workers = len(repl_pool)
        self._worker_pool_queue = asyncio.Queue(maxsize=max_workers)
        self.repl_pool: List[REPL] = repl_pool
        self.stateless = stateless
        
        self._worker_locks = [asyncio.Lock() for _ in repl_pool]

        for worker in self.repl_pool:
            self._worker_pool_queue.put_nowait(worker)

        self._map_lock = asyncio.Lock()
        self._global_env_counter = itertools.count()
        # Map: global_env_id -> (worker_idx, internal_env_id, last_used_timestamp)
        self.env_map: dict[int, tuple[int, int, float]] = {}
        
        self.idle_timeout = repl_pool[0].timeout if repl_pool else IDLE_TIMEOUT
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.debug(f"StateManager created with {max_workers} fully initialized REPL workers.")

    @classmethod
    async def create(cls, repl_config: dict, workers: int = 1, stateless: bool = False) -> "StateManager":
        """Asynchronously creates a StateManager with a pool of initialized REPLs.

        Args:
            repl_config (dict): Configuration for the REPLs.
            workers (int): Number of REPL workers to start.
            stateless (bool): If True, new environments are not saved.

        Returns:
            StateManager: The initialized StateManager instance.
        """

        setup_logger(log_dir=repl_config.get("log_dir", None), log_level=repl_config.get("log_level", None), debug=repl_config.get("debug", False))

        logger.debug(f"Starting StateManager with {workers} workers (eager initialization)...")

        # Instantiate the manager and resolve the config once.
        manager = EnvironmentManager(repl_config)
        resolved_config = manager.resolve_config()
        logger.success(f"Environment resolved. Project Path: {resolved_config['project_path']}")
        repl_config.update(resolved_config) # update the original config

        repl_pool = [REPL(**repl_config) for _ in range(workers)]
        startup_tasks = [repl._start_interactive_async() for repl in repl_pool]
        await asyncio.gather(*startup_tasks)
        if repl_config.get("header"):
            logger.success(f"All {workers} REPL workers have been initialized with the header:\n{repl_config.get('header')}")
        else:
            logger.success(f"All {workers} REPL workers have been initialized.")
        
        instance = cls(repl_pool, stateless=stateless)
        instance.idle_timeout = repl_config.get("idle_timeout", IDLE_TIMEOUT)
        if instance.idle_timeout and instance.idle_timeout > 0:
            instance.start_background_tasks()
        return instance

    async def run(self, command: str, global_env_id: Optional[int] = None) -> Union[Environment, LeanError]:
        """Runs a command in a specific environment or assigns a random available environment.

        Args:
            command (str): The Lean command to execute.
            global_env_id (Optional[int]): The global environment ID to run in.

        Returns:
            Union[Environment, LeanError]: The result of the command.
        """
        if global_env_id is not None:
            async with self._map_lock:
                if global_env_id not in self.env_map:
                    return LeanError(message=f"Unknown global environment ID: {global_env_id}", source="server")
                worker_idx, internal_id, _ = self.env_map[global_env_id]
            worker = self.repl_pool[worker_idx]
            worker_lock = self._worker_locks[worker_idx]
        
        else:
            # No environment ID provided
            worker = await self._worker_pool_queue.get()
            worker_idx = self.repl_pool.index(worker)
            worker_lock = self._worker_locks[worker_idx]
            internal_id = None

        # Acquire the worker's lock
        async with worker_lock:
            result = await worker.run(command, internal_id)
        
        if isinstance(result, Environment):
            if not self.stateless:
                async with self._map_lock:
                    # Create a new global ID for the new state (functional behavior)
                    new_global_id = next(self._global_env_counter)
                    self.env_map[new_global_id] = (worker_idx, result.env, time.time())
                    
                    # Update timestamp for the old global ID to prevent premature cleanup
                    if global_env_id in self.env_map:
                        old_worker, old_internal, _ = self.env_map[global_env_id]
                        self.env_map[global_env_id] = (old_worker, old_internal, time.time())

                result.env = new_global_id
            else:
                # Stateless mode: don't store the environment
                result.env = None

        if global_env_id is None:
            self._worker_pool_queue.put_nowait(worker)
        return result

    async def delete_environment(self, global_env_id: int) -> bool:
        """Deletes an environment by removing it from the map.

        Args:
            global_env_id (int): The global environment ID to delete.

        Returns:
            bool: True if deleted, False if not found.
        """
        async with self._map_lock:
            if global_env_id in self.env_map:
                del self.env_map[global_env_id]
                logger.info(f"Deleted global environment ID: {global_env_id}")
                return True
            else:
                logger.warning(f"Attempted to delete unknown global environment ID: {global_env_id}")
                return False

    def start_background_tasks(self):
        """Starts the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug(f"Started background idle environment cleanup (timeout={self.idle_timeout}s).")

    async def _cleanup_loop(self):
        """Periodically checks for and removes idle environments."""
        while True:
            try:
                await asyncio.sleep(IDLE_CHECK_INTERVAL)
                now = time.time()
                to_delete = []
                async with self._map_lock:
                    for env_id, (_, _, last_used) in self.env_map.items():
                        if now - last_used > self.idle_timeout:
                            to_delete.append(env_id)
                    
                    for env_id in to_delete:
                        del self.env_map[env_id]
                
                if to_delete:
                    logger.debug(f"Cleaned up {len(to_delete)} idle environments: {to_delete}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def shutdown(self):
        """Shuts down all REPLs in the pool."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await asyncio.gather(*(repl.close_async() for repl in self.repl_pool))
        logger.debug("All REPL workers have been shut down.")


