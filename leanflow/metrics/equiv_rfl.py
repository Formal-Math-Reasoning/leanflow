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

import re
from typing import Any, Optional

from loguru import logger

from .base import Metric
from .utils import is_valid_lean, LeanRunner


class EquivRfl(Metric):
    """Computes equivalence by checking for definitional equality using `rfl`. Source: Conjecturing: An Overlooked Step in Formal Mathematical Reasoning (Sivakumar et al., ArXiv 2025)
    
    Example:
        ```python
        from leanflow import EquivRfl

        conjecture_1 = "abbrev foo : Nat := 2"
        conjecture_2 = "abbrev bar : Nat := 1 + 1"

        metric = EquivRfl(repl_config={"lean_version": "4.24.0"})
        result = metric.compute(conjecture_1, conjecture_2)
        print(result)
        ```
    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the EquivRfl metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            **shared_dependencies (Any): Shared dependencies. Must include 'repl_config' or 'client'.
        """
        super().__init__(metric_config, **shared_dependencies)

    def _get_name(self, statement: str) -> Optional[str]:
        """Extracts the definition or theorem name from a Lean statement.

        Args:
            statement (str): The Lean statement.

        Returns:
            (Optional[str]): The extracted name, or None if not found.
        """
        match = re.search(r"\b(?:def|theorem|abbrev|example)\s+([^\s\(\:]+)", statement)
        return match.group(1) if match else None

    def _get_args(self, statement: str) -> Optional[str]:
        """Extracts the arguments of a definition or theorem.

        Args:
            statement (str): The Lean statement.

        Returns:
            (Optional[str]): The extracted arguments string, or None if not found.
        """
        name_match = re.search(r"\b(?:def|theorem|abbrev|example)\s+[^\s\(\:]+", statement)
        if not name_match:
            return None
        
        colon_match = re.search(r"\s*:", statement)
        if not colon_match:
            return None
            
        start = name_match.end()
        end = colon_match.start()
        
        args = statement[start:end].strip()
        return args or None

    async def run_check_async(self, statement_1: str, statement_2: str, header: Optional[str] = None) -> bool:
        """Checks for definitional equality between two statements.

        Args:
            statement_1 (str): The first Lean statement.
            statement_2 (str): The second Lean statement.
            header (Optional[str]): Optional header (imports, etc.) to prepend.

        Returns:
            (bool): True if the statements are definitionally equal, False otherwise.
        """
        
        runner, context = self.get_runner()
        
        async with context:
            context_env = None
            if header is not None:
                try:
                    env_result = await runner.run(header)
                    if not is_valid_lean(env_result):
                        logger.error("The provided header failed to compile for EquivRfl.")
                        return False
                    context_env = env_result.env
                except Exception as e:
                    logger.error(f"An unexpected error occurred during header compilation: {e}")
                    return False

            name1 = self._get_name(statement_1)
            name2 = self._get_name(statement_2)

            if not name1 or not name2:
                logger.error(f"Could not parse name from statements: '{statement_1}' or '{statement_2}'")
                return False

            if name1 == name2:
                new_name2 = name2 + "_2"
                statement_2 = statement_2.replace(name2, new_name2)
                name2 = new_name2

            args = self._get_args(statement_1) or self._get_args(statement_2)
            theorem_args = f"{args} " if args else ""
            call_args = f" {args.split(':')[0].strip('()')} " if args else ""

            template = f"{statement_1}\n\n{statement_2}\n\ntheorem thm {theorem_args}: {name1}{call_args} = {name2}{call_args} := by rfl"

            try:
                result = await runner.run(template, env=context_env)
                return is_valid_lean(result, allow_sorry=False)
            except Exception as e:
                logger.error(f"An unexpected error occurred during rfl check: {e}")
                return False