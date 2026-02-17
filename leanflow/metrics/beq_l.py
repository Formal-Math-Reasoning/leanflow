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

from ..utils import logger
from .base import Metric
from .utils import (
    clean_last_theorem_string, 
    is_valid_lean, 
    check_proof_sub,
)

class BEqL(Metric):
    """Computes the BEq-L metric by checking for bidirectional equivalence using `exact?`.
    Source: Reliable Evaluation and Benchmarks for Statement Autoformalization (Poiroux et al., EMNLP 2025)

    Example:
        ```python
        from leanflow import BEqL

        thm1 = "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry"
        thm2 = "theorem t2 (a b c : Prop) : a → b → c := by sorry"

        metric = BEqL(repl_config={"lean_version": "4.24.0"})
        result = metric.compute(thm1, thm2)
        print(result)
        ```
    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the BEqL metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            **shared_dependencies (Any): Shared dependencies. Must include 'repl_config' or 'client'.
        """
        super().__init__(metric_config, **shared_dependencies)

    async def run_check_async(self, statement_1: str, statement_2: str, header: Optional[str] = None) -> bool:
        """Checks if statement_1 and statement_2 are equivalent using `exact?`.

        Args:
            statement_1 (str): The first Lean statement.
            statement_2 (str): The second Lean statement.
            header (Optional[str]): Optional header (imports, etc.) to prepend.

        Returns:
            (bool): True if both statements can prove each other using `exact?`, False otherwise.
        """
        if statement_1.strip() == statement_2.strip():
            return True

        runner, context = self.get_runner()

        async with context:
            context_env = None
            if header is not None:
                env_result = await runner.run(header)
                if not is_valid_lean(env_result):
                    logger.error("The provided header failed to compile.")
                    return False
                context_env = env_result.env

            base_thm_name = "base_theorem"
            reformulated_thm_name = "reformulated_theorem"

            res = [False, False]
            for i, (base_thm, reform_thm) in enumerate([(statement_1, statement_2), (statement_2, statement_1)]):
                try:
                    formal_1_code = clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True) + "\n\n"
                    formal_2_start_line = formal_1_code.count("\n") + 1
                    formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} := by"
                except ValueError:
                    logger.warning(f"Failed to parse theorem strings during BEqL check (Direction {i}).")
                    continue

                formal_code = formal_1_code + formal_2_code
                
                if await check_proof_sub(runner, formal_code, context_env, formal_2_start_line, "sorry") is None:
                    continue

                # Try to close the goal using `exact?`
                proof_exact = await check_proof_sub(runner, formal_code, context_env, formal_2_start_line, "exact?")
                
                # Verification: The proof must use the base theorem name explicitly
                if proof_exact and base_thm_name in proof_exact:
                    res[i] = True
        
        return all(res)