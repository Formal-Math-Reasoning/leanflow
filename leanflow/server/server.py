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
from omegaconf import OmegaConf

from .manager import StateManager
from ..utils import setup_logger, get_available_cpus, Environment, LeanError

class Server:
    """Server that owns and configures the StateManager.

    Example:
        ```python
        import asyncio
        from leanflow import Server

        async def main():
            server = await Server.create_from_yaml("server.yaml")
            try:
                result = await server.run("#eval 1 + 1")
                print(result)
            finally:
                await server.shutdown()

        asyncio.run(main())
        ```
    """
    def __init__(self, state_manager: StateManager):
        """Initialize the server with a StateManager instance.

        Args:
            state_manager (StateManager): The initialized state manager.
        """
        self.state_manager = state_manager

    @classmethod
    async def create_from_yaml(cls, yaml_path: Union[str, Path]) -> "Server":
        """Asynchronously creates the Server by loading configuration from a YAML file.

        This method waits for the StateManager to be fully initialized before returning.

        Args:
            yaml_path (Union[str, Path]): Path to the YAML configuration file.

        Returns:
            (Server): The initialized Server instance.
        """
        cfg = OmegaConf.load(yaml_path)
        server_cfg = cfg.get("server", {})
        repl_cfg = cfg.get("repl", {})

        setup_logger(log_dir=server_cfg.get("log_dir", None), log_level=server_cfg.get("log_level", None))
        
        repl_config_dict = OmegaConf.to_container(repl_cfg, resolve=True)
        if repl_config_dict.get("project_path") is not None:
            project_path = Path(repl_config_dict["project_path"])
            if not project_path.is_absolute():
                repl_config_dict["project_path"] = str(Path(yaml_path).parent.resolve() / project_path)
        
        workers = server_cfg.get("max_workers", get_available_cpus())
        stateless = server_cfg.get("stateless", False)

        state_manager = await StateManager.create(
            repl_config=repl_config_dict, workers=workers, stateless=stateless
        )
        return cls(state_manager)

    async def run(self, command: str, env: Optional[int] = None) -> Union[Environment, LeanError]:
        """Run command using the StateManager.

        Args:
            command (str): The Lean command to execute.
            env (Optional[int]): The environment ID to run in.

        Returns:
            (Union[Environment, LeanError]): The result of the command.
        """
        return await self.state_manager.run(command, env)

    async def delete_environment(self, env_id: int) -> bool:
        """Delete environment using the StateManager.

        Args:
            env_id (int): The environment ID to delete.

        Returns:
            (bool): True if deleted, False if not found.
        """
        return await self.state_manager.delete_environment(env_id)

    async def shutdown(self):
        """Shutdown the server by shutting down the StateManager."""
        await self.state_manager.shutdown()
