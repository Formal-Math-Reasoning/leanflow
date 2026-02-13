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

from typing import Any, Optional

from .base import Metric
from .utils import is_valid_lean
from ..client import Client

class TypeCheck(Metric):
    """Runs a Lean statement through the REPL and checks for errors during compilation. Returns True when there are no errors returned from Lean. 
    
    Example:
        ```python
        from leanflow import TypeCheck

        metric = TypeCheck(repl_config={"lean_version": "4.24.0"})
        result = metric.compute("theorem test : 1 + 1 = 2 := rfl")
        print(result)
        ```
    """
    
    def __init__(
        self,
        metric_config: dict[str, Any] = {},
        repl_config: dict[str, Any] = {},
        client: Optional[Client] = None,
        **shared_dependencies
    ):
        """Initializes the TypeCheck metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            repl_config (dict[str, Any]): Configuration for the REPL (e.g. {'lean_version': '4.21.0'}).
            client (Optional[Client]): An existing Client instance to use.
            **shared_dependencies (Any): Other shared dependencies.
        """
        super().__init__(metric_config=metric_config, repl_config=repl_config, client=client, **shared_dependencies)

    async def run_check_async(self, statement: str, header: Optional[str] = None) -> bool:
        """Typechecks a Lean statement.
        
        Args:
            statement (str): The Lean code to typecheck.
            header (Optional[str]): Optional header (imports, etc.) to prepend.
            
        Returns:
            (bool): True if the statement typechecks without errors, False otherwise.
        """
        if header is not None:
            statement = f"{header}\n{statement}"
        
        # empty statements do not throw errors but are not valid Lean
        if statement.strip() == "":
            return False

        runner, context = self.get_runner()
        async with context:
            result = await runner.run(statement)
        
        return is_valid_lean(result)