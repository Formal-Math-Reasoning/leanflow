from typing import Any, Optional
import re

from ..utils import logger
from .base import Metric
from .utils import (
    is_valid_lean,
    check_proof_sub,
    clean_last_theorem_string,
    indent_code,
    split_conclusion,
    LeanRunner,
)

class BEqPlus(Metric):
    """Computes the BEq+ metric by checking for bidirectional provability using a variety of tactics.
    Source: Reliable Evaluation and Benchmarks for Statement Autoformalization (Poiroux et al., EMNLP 2025).

    Example:
        ```python
        from leanflow import BEqPlus

        thm1 = "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry"
        thm2 = "theorem t2 (a b c : Prop) : a → b → c := by sorry"

        metric = BEqPlus(repl_config={"lean_version": "4.24.0"})
        result = metric.compute(thm1, thm2, header="import Mathlib")
        print(result)
        ```
    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the BEqPlus metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            **shared_dependencies (Any): Shared dependencies. Must include 'repl_config' or 'client'.
        """
        super().__init__(metric_config, **shared_dependencies)

    async def run_check_async(self, statement_1: str, statement_2: str, header: Optional[str] = None) -> bool:
        """Computes the BEq+ equivalence for the given example.
        
        Args:
            statement_1 (str): The first Lean statement.
            statement_2 (str): The second Lean statement.
            header (Optional[str]): Optional header (imports, etc.) to prepend.

        Returns:
            (bool): True if both statements can prove each other, False otherwise.
        """
        runner, context = self.get_runner()

        async with context:
            context_env = None
            if header is not None:
                env_result = await runner.run(header)
                if not is_valid_lean(env_result):
                    logger.error("The provided header failed to compile.")
                    return False
                context_env = env_result.env

            if statement_1.strip() == statement_2.strip():
                return True

            # Check statement_1 => statement_2
            dir1 = await self._prove_one_direction(runner, statement_1, statement_2, context_env)
            if not dir1:
                return False
                
            # Check statement_2 => statement_1
            dir2 = await self._prove_one_direction(runner, statement_2, statement_1, context_env)
            return dir2

    async def _prove_one_direction(
        self, runner: LeanRunner, base_thm: str, reform_thm: str, context_env: Optional[int]
    ) -> bool:
        """Checks if `base_theorem` implies `reformulated_theorem`."""
        base_thm_name = "base_theorem"
        reformulated_thm_name = "reformulated_theorem"

        def prove_all(tactics: list[str]) -> str:
            prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
            prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
            return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

        solver_tactics_apply = ["tauto", "simp_all!", "noncomm_ring", "exact?"]
        solver_tactics_have = ["tauto", "simp_all!", "exact? using this"]
        
        proof_all_apply = prove_all(solver_tactics_apply)
        proof_all_have = prove_all(solver_tactics_have)
        
        try:
            formal_1_temp = clean_last_theorem_string(base_thm, "temp_placeholder", add_sorry=True)
            formal_2_temp = clean_last_theorem_string(reform_thm, "temp_placeholder", add_sorry=False)
            
            regex_decl = r"(theorem|lemma|example)\s+[^\s:{(\[]+" 
            formal_1_code = re.sub(regex_decl, f"theorem {base_thm_name}", formal_1_temp, count=1) + "\n\n"
            formal_2_cleaned = re.sub(regex_decl, f"theorem {reformulated_thm_name}", formal_2_temp, count=1)
            
            # Prepare the final block for theorem 2: <decl> := by
            if ":=" in formal_2_cleaned:
                 formal_2_code = formal_2_cleaned.split(":=")[0] + " := by"
            else:
                 formal_2_code = formal_2_cleaned + " := by"
            
            # Calculate line number for error checking
            formal_2_start_line = formal_1_code.count("\n") + 1

        except ValueError:
            logger.warning("Failed to parse theorem structure.")
            return False

        formal_code = formal_1_code + formal_2_code
        
        # Sanity Check
        if await check_proof_sub(runner, formal_code, context_env, formal_2_start_line, "sorry") is None:
            return False

        # Try exact?
        proof_exact = await check_proof_sub(runner, formal_code, context_env, formal_2_start_line, "exact?")
        if proof_exact and base_thm_name in proof_exact:
            return True

        # Try apply base_theorem
        proof_apply = await check_proof_sub(
            runner, formal_code, context_env, formal_2_start_line,
            f"apply {base_thm_name}\n" + proof_all_apply
        )
        if proof_apply:
            return True

        # Try 'have' strategy
        check_without = await runner.run(formal_code + "\n" + proof_all_have, env=context_env)
        if is_valid_lean(check_without, allow_sorry=False):
             return True

        idx_conclusion = split_conclusion(formal_1_code)
        if idx_conclusion:
            idx_end_conclusion = formal_1_code.rfind(":=")
            conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
            have_stmt_proof = (
                f"have {conclusion} := by\n"
                + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                + "\n"
            )
            proof_have = await check_proof_sub(
                runner, formal_code, context_env, formal_2_start_line,
                have_stmt_proof + proof_all_have
            )
            if proof_have:
                return True

        # Try convert
        for max_step in range(0, 5):
            proof_convert = await check_proof_sub(
                runner, formal_code, context_env, formal_2_start_line,
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply
            )
            if proof_convert:
                return True
        
        return False