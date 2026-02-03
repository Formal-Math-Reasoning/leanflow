import asyncio
from tqdm.asyncio import tqdm
import re
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI

from .base import BatchMetric
from ..errors import LeanEnvironmentError

class LLMAsAJudge(BatchMetric):
    """Base class for LLM-as-a-judge metrics.
    
    This class provides shared functionality for metrics that use language models
    to evaluate Lean statements, including API and local model support, prompt
    formatting, and result processing.
    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the LLM-as-a-judge metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            **shared_dependencies: Shared dependencies.

        Raises:
            LeanEnvironmentError: If `use_vllm` is True but vLLM is not installed.
        """
        super().__init__(metric_config, **shared_dependencies)
        
        # Verify vLLM installation if needed
        use_vllm = metric_config.get("use_vllm", False)
        if use_vllm:
            try:
                import vllm
            except ImportError:
                raise LeanEnvironmentError("`use_vllm=True` but vLLM is not installed. Install with: pip install leanflow[vllm]")

    def _merge_configs(self, g: Optional[dict], l: Optional[dict]) -> dict:
        """Merge two configuration dictionaries.

        Args:
            g (Optional[dict]): The global configuration dictionary.
            l (Optional[dict]): The local configuration dictionary.
        
        Returns:
            (dict): The merged configuration dictionary.
        """
        p = g.copy() if g else {}
        p.update(l or {})
        return p

    def _get_sampling_params(self, config: dict, is_vllm: bool) -> dict:
        """Get the sampling parameters for the model.

        Args:
            config (dict): The configuration dictionary.
            is_vllm (bool): Whether to use vLLM.
        
        Returns:
            (dict): The sampling parameters.
        """
        if is_vllm:
            try:
                from vllm import SamplingParams
            except ImportError:
                raise LeanEnvironmentError("`use_vllm=True` but vLLM is not installed.")
            vllm_keys = SamplingParams.__fields__.keys()
            vllm_config = {k: v for k, v in config.items() if k in vllm_keys}
            if "stop" not in vllm_config:
                vllm_config["stop"] = []
            # Add default stop tokens (subclasses can override this method to customize)
            self._add_default_stop_tokens(vllm_config)
            return SamplingParams(**vllm_config)
        return {k: v for k, v in config.items() if k in ["temperature", "max_tokens"]}

    def _add_default_stop_tokens(self, vllm_config: dict):
        """Add default stop tokens to vLLM config. Override in subclasses if needed.
        
        Args:
            vllm_config (dict): The vLLM configuration dictionary to modify in-place.
        """
        # Default implementation - subclasses can override
        pass

    def _format_prompt(self, template : dict, **kwargs) -> list[dict]:
        """Format the prompt for the model.

        Args:
            template (dict): The template dictionary.
            **kwargs: The keyword arguments.
        
        Returns:
            (list[dict]): The formatted prompt.
        """
        return [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(**kwargs)}
        ]

    def _remove_lean_comments(self, text: str) -> str:
        """Remove Lean comments from the text.

        Args:
            text (str): The text to remove comments from.
        
        Returns:
            str: The text with comments removed.
        """
        pattern = r"(?:/--|--).*?-/|^\s*--.*"
        return re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE) if text else ""

    def _post_process_comparison(self, text: str) -> Optional[bool]:
        """Post-process the comparison text to extract the boolean value.
        
        This is an abstract method that subclasses should override with their
        specific logic for extracting boolean results from LLM outputs.

        Args:
            text (str): The comparison text.
        
        Returns:
            Optional[bool]: The extracted boolean value, or None if the text is not a valid comparison.
        """
        pass

    def _is_error(self, value: Any) -> bool:
        """Check if the value is an error.

        Args:
            value (Any): The value to check.
        
        Returns:
            bool: True if the value is an error, False otherwise.
        """
        return value is None or (isinstance(value, str) and value.startswith("Error:"))

    async def _run_api_prompts(self, prompts: list[list[dict]], model: str, client: AsyncOpenAI, sampling_params: dict) -> list[str]:
        """Run the prompts on the API model.

        Args:
            prompts (list[list[dict]]): List of prompts to run.
            model (str): The API model to use.
            client (AsyncOpenAI): The client to use.
            sampling_params (dict): The sampling parameters to use.
        
        Returns:
            (list[str]): The responses from the API model.
        """
        max_concurrent = self.metric_config.get("max_concurrent", 10)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _run_single(prompt):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(model=model, messages=prompt, **sampling_params)

                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"API call error with model '{model}'. Error: {e}")
                return f"Error: {e}"

        logger.debug(f"Running batch of {len(prompts)} prompts on API model '{model}'.")
        tasks = [_run_single(p) for p in prompts]
        return await tqdm.gather(*tasks)

    def _run_local_prompts(self, prompts: list[list[dict]], model: str, sampling_params: Any) -> list[str]:
        """Run the prompts on the local model.
        
        Args:
            prompts (list[list[dict]]): List of prompts to run.
            model (str): The local model to use.
            sampling_params (Any): The sampling parameters to use.
        
        Returns:
            (list[str]): The responses from the local model.
        """
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError:
            raise LeanEnvironmentError(f"vLLM and/or transformers not installed, but required for local {self.__class__.__name__}.")

        logger.debug(f"Initializing local vLLM model '{model}'.")
        llm = LLM(model=model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        str_prompts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        
        logger.debug(f"Running batch of {len(str_prompts)} prompts on local model '{model}'.")
        outputs = llm.generate(str_prompts, sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]
