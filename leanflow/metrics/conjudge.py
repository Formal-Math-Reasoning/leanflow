import asyncio
from tqdm.asyncio import tqdm
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI

from .llm_judge_base import LLMAsAJudge
from ..errors import LeanValueError

COMPARISON_TEMPLATE = {
    "system": "You are an expert in the Lean 4 theorem proving language and formal mathematics. Your task is to determine if a given formal statement in Lean 4 contains a specific conjectured value, algebraic formula, or bound.\\n\\nYou will be given three inputs:\\n1.  **Conjecture**: The value, formula, or bound to look for.\\n2.  **Ground Truth Formal Statement**: An example of a Lean 4 statement that correctly formalizes the conjecture. Use this as a reference for a valid implementation.\\n3.  **Formal Statement**: The Lean 4 code you need to evaluate.\\n\\nYour goal is to determine if the **Formal Statement**contains the core assertion of the **Conjecture**. The **Ground Truth Formal Statement** is provided to help you understand how the conjecture can be formally expressed.\\n\\nThe statement you are evaluating might not have the exact same syntax as the ground truth. You must carefully check for **semantically equivalent variations** of the conjecture's core idea. This includes, but is not limited to, permutations of terms, different but equivalent algebraic expressions, or reordered hypotheses. Additionally, a conjecture can be expressed either by defining a proposition (e.g., `abbrev conjecture : Prop := ...`) or by asserting it within a theorem, which implicitly states the conjecture holds. You should consider these forms equivalent.\\n\\nYour output must follow this structure exactly:\\n1.  First, provide a brief explanation of your reasoning.\\n2.  Second, conclude with the final answer in the format: 'The formal statement contains the conjecture: **True**' or 'The formal statement contains the conjecture: **False**'.",
    "user": "**Conjecture:**\\n```lean\\n{conjecture}\\n```\\n\\n**Ground Truth Formal Statement:**\\n```lean\\n{statement1}\\n```\\n\\n**Formal Statement:**\\n```lean\\n{statement2}\\n```"
}

class ConJudge(LLMAsAJudge):
    """Judges if a generated Lean statement correctly incorporates a given conjecture using an LLM. Source: Conjecturing: An Overlooked Step in Formal Mathematical Reasoning (Sivakumar et al., ArXiv 2025)
    
    Example:
        ```python
        from leanflow import ConJudge

        data = {
            "header": "import Mathlib",
            "formal_conjecture": "abbrev conjecture : ℕ : 13",
            "formal_statement": "theorem hackmath_4 : IsLeast {n | ∀ f : Fin n → Fin 12, ∃ a b, f a = f b} ((conjecture) : ℕ ) := by sorry",
            "formal_statement_generated": "theorem hackmath_4 : IsLeast {n | ∀ f : Fin n → Fin 12, ∃ a b, f a = f b} (26 / 2 : ℕ ) := by sorry",
        }

        metric = ConJudge(
            api_config={"base_url": "<URL>", "api_key": "<KEY>"},
            comparison={"model": "gpt-4"}
        )

        result = metric.compute_batch([data])
        print(result)
        ```
    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the ConJudge metric.

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
            getattr(self.__class__, "sampling_params", None) or 
            {}
        )
        
        self.comparison_config = (
            self.metric_config.get("comparison") or 
            getattr(self, "comparison", None) or 
            getattr(self.__class__, "comparison", None) or 
            {}
        )
        
        comp_final_sampling = self._merge_configs(
            global_sampling_params, 
            self.comparison_config.get("sampling_params")
        )
        
        comp_use_vllm = (
            self.comparison_config.get("use_vllm") or
            getattr(self, "use_vllm", None) or
            getattr(self.__class__, "use_vllm", False)
        )
        
        self.comparison_config["sampling_params"] = self._get_sampling_params(
            comp_final_sampling, 
            comp_use_vllm
        )

    def _add_default_stop_tokens(self, vllm_config: dict):
        """Add ConJudge-specific stop tokens to vLLM config.
        
        Args:
            vllm_config (dict): The vLLM configuration dictionary to modify in-place.
        """
        vllm_config["stop"].append("</s>")

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
        true_pos = text_lower.rfind("**true**")
        false_pos = text_lower.rfind("**false**")
        if true_pos > false_pos:
            return True
        if false_pos > true_pos:
            return False
        return None

    async def run_batch_async(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Computes the ConJudge metric for a batch of examples.

        Args:
            examples (list[dict[str, Any]]): List of examples to process.

        Returns:
            (list[dict[str, Any]]): List of results, each containing 'conjudge' (bool) and raw outputs.
        """
        if not examples:
            return []

        results = await self._grade_batch_with_retries(examples)

        return results

    async def _grade_batch_with_retries(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Computes the ConJudge metric for a batch of examples with retries.

        Args:
            examples (list[dict[str, Any]]): List of examples to process.

        Returns:
            (list[dict[str, Any]]): Dictionary of results, each containing 'conjudge' (bool) and raw outputs.
        """
        results = await self._grade_batch_async(examples)
        max_retries = self.metric_config.get("max_retries", 3)

        for i in range(max_retries):
            error_indices = [i for i, r in enumerate(results["conjudge"]) if self._is_error(r)]
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
        """Computes the ConJudge metric for a batch of examples.

        Args:
            examples (list[dict]): List of examples to process.

        Returns:
            (dict[str, list[Any]]): Dictionary of results, each containing 'conjudge' (bool) and raw outputs.
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

        comp_template = self.comparison_config.get("template", COMPARISON_TEMPLATE)
        
        # Remove Lean comments from examples to avoid bias in comments
        formal_conjectures = [self._remove_lean_comments(ex.get("formal_conjecture", "")) for ex in examples]
        s1 = [self._remove_lean_comments(ex.get("formal_statement", "")) for ex in examples]
        s2 = [self._remove_lean_comments(ex.get("formal_statement_generated", "")) for ex in examples]
        
        # replace "conjecture" with "placeholder"
        # s2 = [ex.replace("conjecture", "placeholder") for ex in s2]

        statements1 = [f"{c}\\n\\n{s}" if c else s for c, s in zip(formal_conjectures, s1)]
        
        prompts = [self._format_prompt(template=comp_template, conjecture=c, statement1=s1_full, statement2=s2_single) 
                   for c, s1_full, s2_single in zip(formal_conjectures, statements1, s2)]
        
        raw_comparisons = await _run_prompts(prompts, self.comparison_config)

        return {
            "conjudge_raw": raw_comparisons,
            "conjudge": [self._post_process_comparison(text) for text in raw_comparisons],
        }