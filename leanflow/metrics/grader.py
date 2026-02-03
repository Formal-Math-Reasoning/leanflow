import asyncio
from tqdm.asyncio import tqdm
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI

from .llm_judge_base import LLMAsAJudge
from ..errors import LeanValueError

BACKTRANSLATE_TEMPLATE = {
    "system": "You are a world-class mathematician and an expert in Lean 4. Your sole task is to back-translate a Lean proposition into a precise natural-language math statement. Do not add, remove, or alter assumptions. Do not give a proof or examples. Do not speculate beyond the provided code.",
    "user": "Please translate the following Lean formal statement into a natural language math problem:\\n\\n```lean\\n{formal_statement}\\n```",
}

COMPARISON_TEMPLATE = {
    "system": """You are an expert in mathematical logic and Lean-aware mathematical linguistics. Your sole task is to determine whether two natural-language math statements are semantically equivalent. Focus strictly on logical content: hypotheses, quantifiers and their scope, domains/codomains, and the main conclusion. Do not provide proofs, examples, or restatements beyond what is required.\\n\\nRequirements:\\n- Alpha-renaming (renaming bound variables) is equivalent.\\n- Standard synonymy is equivalent (e.g., 'iff' ↔ 'if and only if'; 'subset' ↔ 'is a subset of'; 'injective' ↔ 'one-to-one'; 'surjective' ↔ 'onto'; 'bijective' ↔ 'one-to-one and onto'; 'ℝ' ↔ 'the real numbers').\\n- Harmless reordering of independent assumptions or conjuncts/disjuncts is equivalent; maintain quantifier scope.\\n- Differences that change meaning include: adding/omitting hypotheses; changing domains/codomains or codomain properties; ∀ vs ∃ (or ∃ vs ∃!); strict vs non-strict inequalities; global vs local quantification; direction of implication; uniqueness claims; total vs partial function assumptions.\\n- Expand implicit context only if it is explicitly present or unambiguously standard in the given text (e.g., 'let G be a group' vs 'G : Group'); otherwise, do not invent assumptions.\\n\\nError handling:\\n- If either input consists only of an error message (e.g., 'ERROR', 'unknown identifier', 'failed to elaborate', compiler/tooling errors), respond exactly: The statements contain an **error**. Then stop.""",
    "user": """Determine whether Statement 1 and Statement 2 are semantically equivalent mathematical statements.\\n\\nOutput format:\\n1) Explanation: a concise comparison focusing on hypotheses and the main claim.\\n2) Decision: exactly one of the following on its own final line:\\nThe statements are **equivalent**.\\nThe statements are **different**.\\n If either statement is only an error message, respond: The statements contain an **error**.\\n\\nStatement 1:\\n{statement_1}\\n\\nStatement 2:\\n{statement_2}"""
}

class LLMGrader(LLMAsAJudge):
    """Computes semantic equivalence using a LLM-as-a-judge. Source: FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models (Yu et al., ArXiv 2025).
    

    This metric performs back-translation of formal statements to natural language using the backtranslation model and then uses the comparison model to compare the semantic equivalence of the back-translated statements.

    Example:
        ```python
        from leanflow import LLMGrader

        data = {
            "formal_statement": "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry",
            "formal_statement_generated": "theorem t2 (a b c : Prop) : a → b → c := by sorry"
        }

        metric = LLMGrader(
            api_config={"base_url": "<URL>", "api_key": "<KEY>"},
            backtranslation={"model": "deepseek-math"},
            comparison={"model": "gpt-4"},
        )

        result = metric.compute_batch([data])
        print(result)
        ```

    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the LLMGrader metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            **shared_dependencies (Any): Shared dependencies.

        Raises:
            LeanEnvironmentError: If `use_vllm` is True but vLLM is not installed.
        """
        super().__init__(metric_config, **shared_dependencies)
        
        self.api_config = getattr(self, "api_config", None)
        
        global_sampling_params = (
            self.metric_config.get("sampling_params") or
            getattr(self, "sampling_params", None) or
            getattr(self.__class__, "sampling_params", {}) or
            {}
        )

        self.backtranslation_config = (
            self.metric_config.get("backtranslation") or
            getattr(self, "backtranslation", None) or
            getattr(self.__class__, "backtranslation", {}) or
            {}
        )
        
        self.comparison_config = (
            self.metric_config.get("comparison") or
            getattr(self, "comparison", None) or
            getattr(self.__class__, "comparison", {}) or
            {}
        )

        bt_final_sampling = self._merge_configs(
            global_sampling_params, 
            self.backtranslation_config.get("sampling_params")
        )
        comp_final_sampling = self._merge_configs(
            global_sampling_params, 
            self.comparison_config.get("sampling_params")
        )
        
        bt_use_vllm = self.backtranslation_config.get("use_vllm") or getattr(self, "use_vllm", False)
        
        comp_use_vllm = self.comparison_config.get("use_vllm") or getattr(self, "use_vllm", False)
        
        self.backtranslation_config["sampling_params"] = self._get_sampling_params(
            bt_final_sampling, 
            bt_use_vllm
        )
        self.comparison_config["sampling_params"] = self._get_sampling_params(
            comp_final_sampling, 
            comp_use_vllm
        )

    def _add_default_stop_tokens(self, vllm_config: dict):
        """Add LLMGrader-specific stop tokens to vLLM config.
        
        Args:
            vllm_config (dict): The vLLM configuration dictionary to modify in-place.
        """
        vllm_config["stop"].append("<|im_end|>")

    def _post_process_comparison(self, text: str) -> Optional[bool]:
        """Post-process the comparison text to extract the boolean value.

        Args:
            text (str): The comparison text.
        
        Returns:
            (Optional[bool]): The extracted boolean value, or None if the text is not a valid comparison.
        """
        if not isinstance(text, str):
            return None
        text_lower = text.lower()
        equivalent_pos = text_lower.rfind("**equivalent**")
        different_pos = text_lower.rfind("**different**")
        if equivalent_pos > different_pos:
            return True
        if different_pos > equivalent_pos:
            return False
        return None

    async def run_batch_async(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Computes the LLM grader metric for a batch of examples.

        Args:
            examples (list[dict[str, Any]]): List of examples to process.

        Returns:
            (list[dict[str, Any]]): List of results, each containing 'llm_grader' (bool) and raw outputs.
        """
        if not examples:
            return []

        results = await self._grade_batch_with_retries(examples)

        return results

    async def _grade_batch_with_retries(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Runs the LLM grader metric with retries for failed samples.

        Args:
            examples (list[dict[str, Any]]): List of examples to process.

        Returns:
            (list[dict[str, Any]]): List of results, each containing 'llm_grader' (bool) and raw outputs.
        """
        results = await self._grade_batch_async(examples)
        max_retries = self.metric_config.get("max_retries", 3)

        for i in range(max_retries):
            error_indices = [i for i, r in enumerate(results["llm_grader"]) if self._is_error(r)]
            if not error_indices:
                break
            
            logger.info(f"Retry {i + 1}/{max_retries} with {len(error_indices)} samples...")
            retry_examples = [examples[i] for i in error_indices]
            retry_results = await self._grade_batch_async(retry_examples)

            for i_retry, original_idx in enumerate(error_indices):
                for key in results:
                    results[key][original_idx] = retry_results[key][i_retry]
            
        list_of_dicts = []
        keys = results.keys()
        for i in range(len(examples)):
            list_of_dicts.append({key: results[key][i] for key in keys})
            
        return list_of_dicts

    async def _grade_batch_async(self, examples: list[dict]) -> dict[str, list[Any]]:
        """Runs the LLM grader metric for a batch of examples.

        Args:
            examples (list[dict]): List of examples to process.

        Returns:
            (dict[str, list[Any]]): Dictionary containing 'llm_grader' (bool) and raw outputs.
        """
        api_client = AsyncOpenAI(**self.api_config) if self.api_config else None

        async def _run_prompts(prompts, config):
            if config.get("use_vllm"):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._run_local_prompts, prompts, config["model"], config["sampling_params"])
            elif api_client:
                return await self._run_api_prompts(prompts, config["model"], api_client, config["sampling_params"])
            else:
                raise LeanValueError(f"Config for model '{config["model"]}' requires `use_vllm=True` or a valid `api_config`.")

        bt_template = self.backtranslation_config.get("template", BACKTRANSLATE_TEMPLATE)
        comp_template = self.comparison_config.get("template", COMPARISON_TEMPLATE)

        # Remove comments from formal statements
        formal_stmts_gold = [self._remove_lean_comments(ex.get("formal_statement")) for ex in examples]
        formal_stmts_gen = [self._remove_lean_comments(ex.get("formal_statement_generated")) for ex in examples]
        unique_gold_stmts = sorted(list(set(s for s in formal_stmts_gold if s)))

        # Batch back-translation
        bt_prompts_unique = [self._format_prompt(bt_template, formal_statement=s) for s in unique_gold_stmts]
        bt_gold_list = await _run_prompts(bt_prompts_unique, self.backtranslation_config)
        
        bt_prompts_gen = [self._format_prompt(bt_template, formal_statement=s) for s in formal_stmts_gen]
        bt_gen = await _run_prompts(bt_prompts_gen, self.backtranslation_config)

        mapping_bt_gold = dict(zip(unique_gold_stmts, bt_gold_list))
        bt_gold = [mapping_bt_gold.get(stmt, "Error: Missing back-translation") for stmt in formal_stmts_gold]

        # Batch comparison
        comp_prompts = [self._format_prompt(comp_template, statement_1=s1, statement_2=s2) for s1, s2 in zip(bt_gold, bt_gen)]
        raw_comparisons = await _run_prompts(comp_prompts, self.comparison_config)

        return {
            "formal_statement_backtranslation": bt_gold,
            "formal_statement_generated_backtranslation": bt_gen,
            "llm_grader_raw": raw_comparisons,
            "llm_grader": [self._post_process_comparison(text) for text in raw_comparisons],
        }