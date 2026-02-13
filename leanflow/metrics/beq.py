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
import re
import os
from tqdm.asyncio import tqdm
from typing import Any, Optional
import concurrent.futures
from openai import AsyncOpenAI

from ..utils import logger
from .base import BatchMetric
from ..repl import REPL
from .utils import is_valid_lean, BEQ_DEFAULT, clean_theorem_string
from ..errors import LeanValueError, LeanEnvironmentError

SYSTEM = "You are an expert in the Lean 4 theorem proving language and formal mathematics."

class BEq(BatchMetric):
    """Computes the BEq metric using a heuristic check followed by LLM-based tactic generation.
    Source: Rethinking and Improving Autoformalization: Towards a Faithful Metric and a Dependency Retrieval-based Approach (Liu et al., 2024)

    First tries a heuristic `exact?` check. If that fails, it uses an LLM in a loop to generate
    and validate tactics to prove equivalence.

    Example:
        ```python
        from leanflow import BEq

        data = {
            "header": "import Mathlib"
            "formal_statement": "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry",
            "formal_statement_generated": "theorem t2 (a b c : Prop) : a → b → c := by sorry",
        }

        metric = BEq(
            api_config={"base_url": "<URL>", "api_key": "<KEY>"},
            tactic_generator={"model": "deepseek-math"},
            repl_config={"lean_version": "v4.24.0"}
        )

        result = metric.compute_batch([data])
        print(result)
        ```

    """

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        """Initializes the BEq metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary.
            **shared_dependencies (Any): Shared dependencies. Must include 'repl_config' or 'client'.

        Raises:
            LeanValueError: If neither 'repl_config' nor 'client' is provided, or if 'tactic_generator.model' is missing.
        """
        super().__init__(metric_config, **shared_dependencies)
        
        self.api_config = getattr(self, "api_config", None)

        if not self.repl_config and not self.client:
            raise LeanValueError("BEq metric requires 'repl_config' or a 'client' instance.")
        
        self.use_mp = self.metric_config.get("use_mp", True) and self.client is None and self.repl_config is not None
        
        self.try_num = self.metric_config.get("try_num", 8)
        self.timeout = self.metric_config.get("timeout", 300)
        beq_setting = self.metric_config.get("beq_setting", "all")
        if beq_setting not in BEQ_DEFAULT: 
            raise LeanValueError(f"Unknown beq_setting: {beq_setting}")
        self.allowed_tactics, self.banned_tokens, self.template = BEQ_DEFAULT[beq_setting].values()

        self.tactic_gen_config = self.metric_config.get("tactic_generator", self.__dict__.get("tactic_generator", {}))
        self.model_path = self.tactic_gen_config.get("model")
        if not self.model_path:
            raise LeanValueError("BEq metric requires a 'tactic_generator.model' path.")
        
        sampling_params = self._merge_configs(self.metric_config.get("sampling_params"), self.tactic_gen_config.get("sampling_params"))
        
        self.use_beam_search = self.metric_config.get("use_beam_search", False)
        
        default_temp = 0.0 if self.try_num == 1 else 0.7
        self.temperature = self.metric_config.get("temperature", default_temp)
        
        self.sampling_params = self._get_sampling_params(sampling_params)

    async def run_batch_async(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Computes the BEq metric for a batch of examples.

        Args:
            examples (list[dict[str, Any]]): List of examples to process.

        Returns:
            (list[dict[str, Any]]): List of results, each containing 'beq' (bool) and 'generated_tactics'.
        """
        if not examples:
            return []
        
        self._preprocess_examples(examples)
        
        num_examples = len(examples)
        beq_results = [False] * num_examples
        beq_tactics = [[] for _ in range(num_examples)]
        
        # Track success for each direction independently
        llm_pq_success = [False] * num_examples
        llm_qp_success = [False] * num_examples

        # Heuristic Check
        logger.info(f"Running heuristic `exact?` check on {num_examples} examples.")
        heur_func = self._heuristic_check_async if self.client else self._heuristic_check_entrypoint_sync
        heuristic_results = await self._run_checks_parallel(heur_func, examples, list(range(num_examples)))
        
        for i, success in enumerate(heuristic_results):
            beq_results[i] = success
            # If heuristic passes, both directions are considered solved via "exact?"
            if success:
                llm_pq_success[i] = True
                llm_qp_success[i] = True
            beq_tactics[i].append({"tactic": "exact?", "success": success, "direction": "both"})

        # LLM Retry Loop
        for i_try in range(self.try_num):
            indices_to_retry = [i for i, success in enumerate(beq_results) if not success]
            
            if not indices_to_retry:
                logger.info(f"All examples solved. Halting LLM generation.")
                break
            
            logger.info(f"Try {i_try + 1}/{self.try_num}: generating tactics for pending directions.")

            prompts, task_meta = self._prepare_sparse_llm_prompts(examples, indices_to_retry, llm_pq_success, llm_qp_success)
            
            if not prompts:
                break

            # Batch Generation
            generated_texts = await self._llm_generate_tactics(prompts)
                        
            gen_map = {}
            for meta, text in zip(task_meta, generated_texts):
                idx = meta["idx"]
                direction = meta["direction"]
                if idx not in gen_map: gen_map[idx] = {}
                gen_map[idx][direction] = text

            check_data_list = []
            for idx in indices_to_retry:
                # We pass the generated tactic if available, otherwise None
                tactic_pq = gen_map.get(idx, {}).get("PQ", None)
                tactic_qp = gen_map.get(idx, {}).get("QP", None)
                
                # We only need to validate if we actually generated something new
                if tactic_pq is not None or tactic_qp is not None:
                    check_data_list.append({
                        "original_idx": idx,
                        "generated_PQ": tactic_pq, 
                        "generated_QP": tactic_qp,
                        "skip_pq": llm_pq_success[idx],
                        "skip_qp": llm_qp_success[idx]
                    })

            # Validation
            llm_func = self._llm_check_async if self.client else self._llm_check_entrypoint_sync
            validation_results = await self._run_checks_parallel(llm_func, examples, check_data_list)

            # Update State
            for i, (res_pq, res_qp) in enumerate(validation_results):
                original_idx = check_data_list[i]["original_idx"]
                
                if res_pq:
                    success, tactic = res_pq
                    if success: llm_pq_success[original_idx] = True
                    if tactic: 
                        beq_tactics[original_idx].append({"tactic": tactic, "success": success, "direction": "PQ"})
                
                # Process QP result
                if res_qp:
                    success, tactic = res_qp
                    if success: llm_qp_success[original_idx] = True
                    if tactic:
                        beq_tactics[original_idx].append({"tactic": tactic, "success": success, "direction": "QP"})

                # Check if full equivalence is now met
                if llm_pq_success[original_idx] and llm_qp_success[original_idx]:
                    beq_results[original_idx] = True

        return [{"beq": beq_results[i], "generated_tactics": beq_tactics[i]} for i in range(num_examples)]

    # Methods for Parallel Execution
    async def _run_checks_parallel(self, func, all_examples, items_to_process):

        is_async = asyncio.iscoroutinefunction(func)
        
        if self.client and is_async:
            tasks = [func(all_examples, item, self.client) for item in items_to_process]
            return await tqdm.gather(*tasks)
        elif self.use_mp and not is_async:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ProcessPoolExecutor() as pool:
                futures = [loop.run_in_executor(pool, func, all_examples, item, None) for item in items_to_process]
                return await tqdm.gather(*futures)
        elif is_async:
            return [await func(all_examples, item, None) for item in items_to_process]
        else:
            # sync function without multiprocessing (shouldn't happen)
            return [func(all_examples, item, None) for item in items_to_process]

    # Sync Entrypoints for Multiprocessing
    def _heuristic_check_entrypoint_sync(self, all_examples, idx, runner_override):
        return asyncio.run(self._heuristic_check_async(all_examples, idx, runner_override))
    
    def _llm_check_entrypoint_sync(self, all_examples, check_data, runner_override):
        return asyncio.run(self._llm_check_async(all_examples, check_data, runner_override))

    # Async Logic for REPL/Client Interaction
    async def _heuristic_check_async(self, all_examples, idx, runner_override):
        ex = all_examples[idx]
        res_pq = await self._repl_heuristic_exact_async(ex["header"], ex["code_PQ_P"], ex["code_PQ_Q"], runner_override)
        if not res_pq: return False
        return await self._repl_heuristic_exact_async(ex["header"], ex["code_QP_P"], ex["code_QP_Q"], runner_override)

    async def _llm_check_async(self, all_examples, check_data, runner_override):
        ex = all_examples[check_data["original_idx"]]
        
        tasks = []
        
        if check_data.get("generated_PQ") and not check_data.get("skip_pq"):
            tasks.append(self._llm_check_equivalence_async(
                ex["header"], ex["code_PQ_P"], ex["code_PQ_Q"], check_data["generated_PQ"], runner_override
            ))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        # QP Check
        if check_data.get("generated_QP") and not check_data.get("skip_qp"):
            tasks.append(self._llm_check_equivalence_async(
                ex["header"], ex["code_QP_P"], ex["code_QP_Q"], check_data["generated_QP"], runner_override
            ))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        return await asyncio.gather(*tasks)

    async def _repl_heuristic_exact_async(self, header, code_p, code_q, runner_override):
        runner = self._get_runner(runner_override)
        if runner_override:
            try:
                env = await runner.run(header)
                if not code_q.endswith(":= by"): return False
                output = await runner.run(f"{code_p}\n\n{code_q}\nexact?\n", env=env)
                if not is_valid_lean(output, allow_sorry=True): return False
                return any("Try this: exact" in m.data and "thm_P" in m.data for m in getattr(output, "messages", []))
            except Exception: return False
        else:
            async with runner:
                try:
                    env = await runner.run(header)
                    if not code_q.endswith(":= by"): return False
                    output = await runner.run(f"{code_p}\n\n{code_q}\nexact?\n", env=env)
                    if not is_valid_lean(output, allow_sorry=True): return False
                    return any("Try this: exact" in m.data and "thm_P" in m.data for m in getattr(output, "messages", []))
                except Exception: return False

    async def _llm_check_equivalence_async(self, header, code_p, code_q, proof, runner_override):
        tactic = self._extract_first_code_block(proof)
        if any(t in tactic for t in self.banned_tokens) or (self.allowed_tactics and not any(t in tactic for t in self.allowed_tactics)):
            return False, tactic
        runner = self._get_runner(runner_override)
        if runner_override:
            try:
                env = await runner.run(header)
                output = await runner.run(f"{code_p}\n\n{code_q}\n{tactic}\n", env=env)
                if not is_valid_lean(output, allow_sorry=False): return False, tactic
                thm_p_used = "thm_P" in tactic or any("Try this: exact" in m.data and "thm_P" in m.data for m in getattr(output, 'messages', []))
                return thm_p_used, tactic
            except Exception: return False, tactic
        else:
            async with runner:
                try:
                    env = await runner.run(header)
                    output = await runner.run(f"{code_p}\n\n{code_q}\n{tactic}\n", env=env)
                    if not is_valid_lean(output, allow_sorry=False): return False, tactic
                    thm_p_used = "thm_P" in tactic or any("Try this: exact" in m.data and "thm_P" in m.data for m in getattr(output, 'messages', []))
                    return thm_p_used, tactic
                except Exception: return False, tactic

    def _get_runner(self, runner_override):
        if runner_override:
            return runner_override
        return REPL(**self.repl_config)
    
    def _preprocess_examples(self, examples):
        for ex in examples:
            s_gen, h_gen = self._prepare_statement(ex["formal_statement_generated"], "thm_Q", ex.get("header", ""))
            s_gt, _ = self._prepare_statement(ex["formal_statement"], "thm_P", ex.get("header", ""))
            ex["header"], ex["code_PQ_P"], ex["code_PQ_Q"] = h_gen, s_gt, self._insert_by(s_gen)
            ex["code_QP_P"], ex["code_QP_Q"] = s_gen.replace("thm_Q", "thm_P"), self._insert_by(s_gt.replace("thm_P", "thm_Q"))

    # def _prepare_llm_prompts(self, examples, indices):
    #     prompts_pq, prompts_qp = [], []
    #     use_vllm = self.tactic_gen_config.get("use_vllm", False)
    #     tokenizer = self._get_tokenizer() if use_vllm else None
    #     for i in indices:
    #         ex = examples[i]
    #         pq_content = self.template.replace('{autoformalization_result}', f"{ex['header']}\n{ex['code_PQ_P']}\n\n{ex['code_PQ_Q']}")
    #         prompts_pq.append(self._format_prompt(pq_content, tokenizer, use_vllm))
    #         qp_content = self.template.replace('{autoformalization_result}', f"{ex['header']}\n{ex['code_QP_P']}\n\n{ex['code_QP_Q']}")
    #         prompts_qp.append(self._format_prompt(qp_content, tokenizer, use_vllm))
    #     return prompts_pq, prompts_qp
    
    def _prepare_sparse_llm_prompts(self, examples, indices, success_pq, success_qp):
        """Generates prompts only for the directions that have not yet succeeded."""
        prompts = []
        task_meta = [] # Stores {'idx': int, 'direction': 'PQ'|'QP'}
        
        use_vllm = self.tactic_gen_config.get("use_vllm", False)
        tokenizer = self._get_tokenizer() if use_vllm else None

        for i in indices:
            ex = examples[i]
            
            # Only generate PQ if it hasn't succeeded yet
            if not success_pq[i]:
                pq_content = self.template.replace("{autoformalization_result}", f"{ex['header']}\n{ex['code_PQ_P']}\n\n{ex['code_PQ_Q']}")
                prompts.append(self._format_prompt(pq_content, tokenizer, use_vllm))
                task_meta.append({"idx": i, "direction": "PQ"})
            
            # Only generate QP if it hasn't succeeded yet
            if not success_qp[i]:
                qp_content = self.template.replace("{autoformalization_result}", f"{ex['header']}\n{ex['code_QP_P']}\n\n{ex['code_QP_Q']}")
                prompts.append(self._format_prompt(qp_content, tokenizer, use_vllm))
                task_meta.append({"idx": i, "direction": "QP"})
                
        return prompts, task_meta

    async def _llm_generate_tactics(self, prompts: list):
        if self.tactic_gen_config.get("use_vllm"):
            from vllm import LLM

            model = LLM(self.model_path, tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")), trust_remote_code=True)
            fn_generate = model.beam_search if self.metric_config.get("use_beam_search") else model.generate
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: [r.outputs[0].text for r in fn_generate(prompts, self.sampling_params)])
        else:
            client = AsyncOpenAI(**self.api_config)
            max_concurrent = self.tactic_gen_config.get("max_concurrent", 10)
            
            async def _generate(msg):
                try:
                    res = await client.chat.completions.create(model=self.model_path, messages=msg, **self.sampling_params)
                    return res.choices[0].message.content.strip()
                except Exception as e:
                    return f"Error: {e}"

            tasks = [_generate(p) for p in prompts]
            return await tqdm.gather(*tasks)

    async def _run_in_pool(self, func, items: list):
        if not self.use_mp:
            return [func(item) for item in items]
        
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            futures = [loop.run_in_executor(pool, func, item) for item in items]
            results = await tqdm.gather(*futures)
        return results

    def _wrapper_repl_heuristic_exact(self, idx: int):
        ex = self.examples[idx]
        res_pq = self._repl_heuristic_exact(ex["header"], ex["code_PQ_P"], ex["code_PQ_Q"])
        if not res_pq: 
            return False
        return self._repl_heuristic_exact(ex["header"], ex["code_QP_P"], ex["code_QP_Q"])

    def _wrapper_llm_check_equivalence(self, check_data: dict):
        ex = self.examples[check_data["original_idx"]]
        res_pq = self._llm_check_equivalence(ex["header"], ex["code_PQ_P"], ex["code_PQ_Q"], check_data["generated_PQ"])
        res_qp = self._llm_check_equivalence(ex["header"], ex["code_QP_P"], ex["code_QP_Q"], check_data["generated_QP"])
        return res_pq, res_qp

    def _repl_heuristic_exact(self, header: str, code_p: str, code_q: str) -> bool:
        try:
            with REPL(**self.repl_config) as repl:
                env = repl.run(header)
                if not code_q.endswith(":= by"):
                    return False
                output = repl.run(f"{code_p}\n\n{code_q}\nexact?\n", env=env)
                if not is_valid_lean(output, allow_sorry=True): return False
                return any("Try this: exact" in m.data and "thm_P" in m.data for m in output.messages)
        except Exception:
            return False

    def _llm_check_equivalence(self, header: str, code_p: str, code_q: str, proof: str) -> "tuple[bool, str]":
        tactic = self._extract_first_code_block(proof)
        if any(t in tactic for t in self.banned_tokens) or (self.allowed_tactics and not any(t in tactic for t in self.allowed_tactics)):
            return False, tactic
        try:
            with REPL(**self.repl_config) as repl:
                env = repl.run(header)
                output = repl.run(f"{code_p}\n\n{code_q}\n{tactic}\n", env=env)
                if not is_valid_lean(output, allow_sorry=False): 
                    return False, tactic
                thm_p_used = "thm_P" in tactic or any("Try this: exact" in m.data and "thm_P" in m.data for m in output.messages)
                return thm_p_used, tactic
        except Exception:
            return False, tactic

    def _merge_configs(self, global_params: Optional[dict], local_params: Optional[dict]) -> dict:
        params = global_params.copy() if global_params else {}
        if local_params: params.update(local_params)
        return params

    def _get_sampling_params(self, config: dict):
        if self.tactic_gen_config.get("use_vllm", False):
            try:
                from vllm.sampling_params import SamplingParams
            except ImportError:
                raise LeanEnvironmentError("vLLM is required for local BEq tactic generation.")
            # Create a copy to avoid mutating the original
            vllm_config = dict(config)
            # Ensure stop is a list - vLLM requires this
            if "stop" not in vllm_config:
                vllm_config["stop"] = ["<|im_end|>"]
            elif isinstance(vllm_config.get("stop"), str):
                vllm_config["stop"] = [vllm_config["stop"]]
            elif not isinstance(vllm_config.get("stop"), list):
                vllm_config["stop"] = ["<|im_end|>"]
            # Ensure temperature is set if not in config
            if "temperature" not in vllm_config:
                vllm_config["temperature"] = self.temperature
            # Filter out model parameter
            return SamplingParams(**{k: v for k, v in vllm_config.items() if k != "model"})
        
        # For OpenAI/API
        params = {k: v for k, v in config.items() if k in ["temperature", "top_p", "max_tokens"]}
        if "temperature" not in params:
            params["temperature"] = self.temperature
        return params

    def _get_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            model_name = self.tactic_gen_config.get("hf_model_name", self.model_path)
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except ImportError:
            logger.warning("`transformers` is not installed. Cannot get tokenizer for local vLLM model.")
            return None

    def _format_prompt(self, content, tokenizer, use_vllm):
        if use_vllm:
            # FIX: Use apply_chat_template if available for correct model formatting
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content}
                ]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Fallback for models without chat template in tokenizer
            return f"{SYSTEM}\n\nUser: {content}\n\nAssistant:"
        
        # API handles list of dicts naturally
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content}
        ]

    def _extract_first_code_block(self, text: str) -> str:
        match = re.search(r"```(?:\w*)?\s*([\s\S]*?)(?:```|$)", text, re.DOTALL)
        return match.group(1).strip() if match else text

    def _prepare_statement(self, statement: str, new_name: str, header: str) -> "tuple[str, str]":
        """Prepares the theorem statement by cleaning and renaming it."""
        clean_stmt = clean_theorem_string(statement, new_theorem_name=new_name, add_sorry=False)
        if clean_stmt is None:
            clean_stmt = re.sub(r"theorem\s+\S+", f"theorem {new_name}", statement, count=1)
        return clean_stmt, header

    def _insert_by(self, statement: str) -> str:
        """Inserts ':= by' at the end of the statement, replacing 'sorry'."""
        return re.sub(r":=(\s*(by)*\n*)*sorry", ":= by", statement).strip()