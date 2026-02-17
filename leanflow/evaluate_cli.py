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

"""
Evaluation CLI for LeanFlow - Run metrics on autoformalization datasets.

This module provides the command-line interface for evaluating autoformalization
models using the LeanFlow framework. It supports both interactive (local) and
server-based Lean execution modes.

Example:
    python -m leanflow.evaluate_cli --config config.yaml
"""

import asyncio
import concurrent.futures
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from datasets import load_from_disk, load_dataset, Dataset
from typing import Any, Optional
import functools
import argparse
import sys
import os
import re
from tqdm.asyncio import tqdm
import importlib.util
from pathlib import Path

from .client import Client
from .utils import logger, setup_logger
from .errors import LeanValueError
from .metrics import (
    Metric,
    BatchMetric,
    TypeCheck,
    BEqL,
    BEqPlus,
    EquivRfl,
    BEq,
    LLMGrader,
    ConJudge
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_COLUMN_NAMES = {
    "formal_statement": "formal_statement",
    "formal_statement_generated": "formal_statement_generated",
    "formal_conjecture": "formal_conjecture",
    "formal_conjecture_generated": "formal_conjecture_generated",
    "header": "header"
}

BUILT_IN_METRICS: dict[str, type[Metric]] = {
    "typecheck": TypeCheck,
    "beq_l": BEqL,
    "beq_plus": BEqPlus,
    "equiv_rfl": EquivRfl,
    "beq": BEq,
    "llm_grader": LLMGrader,
    "conjudge": ConJudge,
}

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(data_path: str) -> list[dict[str, Any]]:
    """Load dataset from disk.
    
    Supports HuggingFace datasets (directories) and JSON files.
    
    Args:
        data_path: Path to dataset file or directory
        
    Returns:
        List of example dictionaries
        
    Raises:
        FileNotFoundError: If the specified path doesn't exist
        ValueError: If the file format is not supported
    """
    path = Path(data_path)
    
    if path.is_dir():
        logger.debug(f"Loading HuggingFace dataset from directory: {path}")
        dataset = load_from_disk(str(path))
    elif path.is_file() and path.suffix == ".json":
        logger.debug(f"Loading JSON dataset from file: {path}")
        dataset = load_dataset("json", data_files=str(path))["train"]
    else:
        raise LeanValueError(
            f"Dataset not found or unsupported format: {data_path}\n"
            f"Supported formats: HuggingFace dataset directory or .json file"
        )
    
    return list(dataset)


def preprocess_data(
    examples: list[dict[str, Any]],
    config: DictConfig,
    column_names: dict[str, str]
) -> list[dict[str, Any]]:
    """Preprocess headers and statements based on configuration.
    
    Supports three modes:
    - set_header: Replace all headers with a custom header
    - remove_header: Remove all headers from code
    - extract_header: Extract headers into separate field
    
    Args:
        examples: List of example dictionaries
        config: Configuration object with preprocessing options
        column_names: Dictionary mapping logical names to actual column names
        
    Returns:
        Preprocessed examples
    """
    logger.debug("Starting data preprocessing...")
    header_pattern = r"(?:^(?:import|open|set_option)[^\n]*\n)+"
    
    formal_stmt = column_names["formal_statement"]
    formal_stmt_gen = column_names["formal_statement_generated"]
    
    if "set_header" in config:
        new_header = config.set_header
        logger.info(f"Setting custom header for all statements")
        for ex in examples:
            for key in [formal_stmt, formal_stmt_gen]:
                if key in ex and ex[key]:
                    code = re.sub(header_pattern, "", ex[key]).strip()
                    ex[key] = f"{new_header}\n{code}"
    
    elif config.get("remove_header", False):
        logger.info("Removing headers from all statements")
        for ex in examples:
            for key in [formal_stmt, formal_stmt_gen]:
                if key in ex and ex[key]:
                    ex[key] = re.sub(header_pattern, "", ex[key]).strip()

    elif config.get("extract_header", False):
        logger.info(f"Extracting headers from '{formal_stmt_gen}'")
        for ex in examples:
            if formal_stmt_gen in ex and ex[formal_stmt_gen]:
                match = re.search(header_pattern, ex[formal_stmt_gen])
                header = match.group(0) if match else ""
                code = re.sub(header_pattern, "", ex[formal_stmt_gen]).strip()
                ex[formal_stmt_gen] = code
                ex["header_generated"] = header
    
    return examples

# =============================================================================
# Metric Loading and Management
# =============================================================================

def load_metric_class(metric_name: str, config: DictConfig) -> type[Metric]:
    """Load a metric class from built-in metrics or custom source file.
    
    Args:
        metric_name: Name of the metric to load
        config: Configuration object containing metric settings
        
    Returns:
        Metric class
        
    Raises:
        FileNotFoundError: If custom source file doesn't exist
        ImportError: If module cannot be loaded
        AttributeError: If class not found in module
        ValueError: If metric is not built-in and no source_file provided
    """
    metric_config = config.get(metric_name) or {}

    if "source_file" in metric_config:
        # Load custom metric from source file
        source_path = Path(metric_config.source_file)
        class_name = metric_config.get(
            "class_name",
            metric_name.replace("_", " ").title().replace(" ", "")
        )
        
        if not source_path.is_file():
            raise LeanValueError(f"Custom metric source file not found: {source_path}")
        
        logger.debug(f"Loading custom metric '{metric_name}' from {source_path}")
        spec = importlib.util.spec_from_file_location(f"custom_metric.{metric_name}", source_path)
        
        if spec is None or spec.loader is None:
            raise LeanValueError(f"Could not create module spec for {source_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, class_name):
            raise LeanValueError(f"Class '{class_name}' not found in '{source_path}'")
        
        return getattr(module, class_name)

    elif metric_name in BUILT_IN_METRICS:
        # Load built-in metric
        logger.debug(f"Loading built-in metric '{metric_name}'")
        return BUILT_IN_METRICS[metric_name]
    
    else:
        # Unknown metric without source file
        raise LeanValueError(
            f"Metric '{metric_name}' is not built-in and no 'source_file' was provided.\n"
            f"Built-in metrics: {', '.join(BUILT_IN_METRICS.keys())}\n"
            f"To use a custom metric, add 'source_file' to its config."
        )


def example_to_metric_args(
    metric_name: str,
    example: dict[str, Any],
    config: DictConfig,
    column_names: dict[str, str]
) -> tuple[Any, ...]:
    """Convert an example dictionary to metric-specific arguments.
    
    Each metric has its own signature. This function maps example fields
    to the appropriate arguments for each metric type.
    
    Args:
        metric_name: Name of the metric
        example: Example dictionary from dataset
        config: Configuration object
        column_names: Dictionary mapping logical names to actual column names
        
    Returns:
        Tuple of arguments to pass to metric's run_check_async()
    """
    metric_config = config.get(metric_name) or {}
    
    formal_stmt = column_names["formal_statement"]
    formal_stmt_gen = column_names["formal_statement_generated"]
    formal_conj = column_names["formal_conjecture"]
    formal_conj_gen = column_names["formal_conjecture_generated"]
    header_col = column_names["header"]
    
    if metric_name == "typecheck":
        statement = example.get(formal_stmt, "")
        header = metric_config.get(header_col) or example.get(header_col)
        return (statement, header) if header else (statement,)
    
    elif metric_name in ["beq_plus", "beq_l"]:
        statement_1 = example.get(formal_stmt, "")
        statement_2 = example.get(formal_stmt_gen, "")
        header = metric_config.get(header_col) or example.get(header_col)
        return (statement_1, statement_2, header) if header else (statement_1, statement_2)
    
    elif metric_name == "equiv_rfl":
        conj_1 = example.get(formal_conj, "")
        conj_2 = example.get(formal_conj_gen, "")
        return (conj_1, conj_2)
    
    else:
        # Default: pass the whole example dictionary
        return (example,)


def run_stateless_metric_in_process(
    metric_class: type[Metric],
    metric_config: dict[str, Any],
    shared_deps: dict[str, Any],
    example: dict[str, Any],
    metric_name: str,
    global_config: DictConfig,
    column_names: dict[str, str]
) -> Any:
    """Execute a stateless metric in a separate process for multiprocessing.
    
    This function is designed to be pickled and executed in a worker process.
    
    Args:
        metric_class: The metric class to instantiate
        metric_config: Configuration for this specific metric
        shared_deps: Shared dependencies (repl_config, etc.)
        example: The example to evaluate
        metric_name: Name of the metric (for argument mapping)
        global_config: Global configuration object
        column_names: Dictionary mapping logical names to actual column names
        
    Returns:
        Metric evaluation result
    """
    try:
        metric_instance = metric_class(metric_config, **shared_deps)
        args = example_to_metric_args(metric_name, example, global_config, column_names)
        result = metric_instance.compute(*args)
        return result
    except Exception as e:
        logger.error(f"Metric '{metric_name}' failed on example: {e}")
        return {"error": str(e)}

# =============================================================================
# Results Management
# =============================================================================

def save_and_log_results(
    df: pd.DataFrame,
    metric: str,
    results: list[Any]
) -> pd.DataFrame:
    """Add metric results to dataframe and log summary statistics.
    
    Args:
        df: Results dataframe
        metric: Name of the metric
        results: List of metric results
        
    Returns:
        Updated dataframe
    """
    # Add results to dataframe
    for i, res in enumerate(results):
        if isinstance(res, dict):
            # For metrics that return dictionaries (e.g., with multiple fields)
            for key, value in res.items():
                df.loc[i, f"{metric}_{key}"] = value
        else:
            # For metrics that return single values
            df.loc[i, metric] = res
    
    # Log summary statistics
    try:
        if metric in df.columns:
            avg = df[metric].mean()
            pos = df[metric].sum()
            total = len(df[metric])
            logger.success(
                f"\n====================\n"
                f"Metric: '{metric}'\n"
                f"Results: {avg:.2f} ({pos}/{total})\n"
                f"===================="
            )
    except Exception as e:
        logger.warning(f"Could not log results for '{metric}': {e}")

    return df

# =============================================================================
# Configuration and Setup
# =============================================================================

def setup_environment(config: DictConfig):
    """Setup logging and CUDA environment from configuration.
    
    Args:
        config: Configuration object
    """
    setup_logger(
        log_dir=config.get("log_dir", None),
        log_level=config.get("log_level", None)
    )
    
    if "devices" in config:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.devices)
        logger.debug(f"Set CUDA_VISIBLE_DEVICES={config.devices}")


def setup_runner(config: DictConfig) -> tuple[dict[str, Any], Optional[Client]]:
    """Setup shared dependencies and client/repl configuration.
    
    Determines whether to use interactive (local) or server mode based on
    the presence of 'base_url' in repl_config.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (shared_deps, client)
        - shared_deps: Dictionary with api_config, sampling_params, and either client or repl_config
        - client: Client instance if using server mode, None otherwise
    """
    shared_deps = {
        "api_config": config.get("api_config"),
        "sampling_params": config.get("sampling_params")
    }

    client = None
    if "repl_config" in config:
        if "base_url" in config.repl_config:
            # Server mode
            logger.debug(f"REPL Mode: Server. Initializing client for {config.repl_config.base_url}")
            client = Client(**config.repl_config)
            shared_deps["client"] = client
        else:
            # Interactive mode
            logger.debug("REPL Mode: Interactive. `repl_config` will be passed to metrics.")
            shared_deps["repl_config"] = config.repl_config
    
    return shared_deps, client


def load_metrics(
    config: DictConfig,
    results_df: pd.DataFrame,
    shared_deps: dict[str, Any]
) -> tuple[dict[str, type[Metric]], dict[str, BatchMetric]]:
    """Load and categorize metrics into stateless and batch metrics.
    
    Args:
        config: Configuration object
        results_df: Results dataframe (to check for existing metrics)
        shared_deps: Shared dependencies for metric initialization
        
    Returns:
        Tuple of (stateless_metrics, batch_metrics)
        - stateless_metrics: Dict mapping metric names to metric classes
        - batch_metrics: Dict mapping metric names to instantiated batch metrics
    """
    stateless_metrics: dict[str, type[Metric]] = {}
    batch_metrics: dict[str, BatchMetric] = {}
    
    for metric_name in config.get("metrics", []):
        metric_conf = config.get(metric_name, OmegaConf.create({}))
        
        # Skip if metric already exists and overwrite is not enabled
        if (metric_name in results_df.columns and
            not config.get("overwrite_all") and
            not metric_conf.get("overwrite")):
            logger.info(f"Skipping metric '{metric_name}' as it already exists")
            continue
        
        # Load metric class
        metric_class = load_metric_class(metric_name, config)
        
        # Categorize as batch or stateless metric
        if issubclass(metric_class, BatchMetric):
            batch_metrics[metric_name] = metric_class(metric_conf, **shared_deps)
            logger.debug(f"Loaded batch metric: {metric_name}")
        else:
            stateless_metrics[metric_name] = metric_class
            logger.debug(f"Loaded stateless metric: {metric_name}")
    
    return stateless_metrics, batch_metrics

# =============================================================================
# Metric Execution
# =============================================================================

async def run_stateless_metrics(
    metrics: dict[str, type[Metric]],
    examples: list[dict[str, Any]],
    config: DictConfig,
    shared_deps: dict[str, Any],
    client: Optional[Client],
    results_df: pd.DataFrame,
    column_names: dict[str, str]
) -> pd.DataFrame:
    """Execute stateless metrics with multiprocessing or async execution.
    
    Uses multiprocessing for interactive mode (better parallelization of
    separate REPL instances) and async execution for server mode (shares
    a single client connection).
    
    Args:
        metrics: Dictionary of metric names to metric classes
        examples: List of examples to evaluate
        config: Configuration object
        shared_deps: Shared dependencies
        client: Client instance (if server mode) or None
        results_df: Results dataframe to update
        column_names: Dictionary mapping logical names to actual column names
        
    Returns:
        Updated results dataframe
    """
    if not metrics:
        return results_df
    
    logger.info(f"Running {len(metrics)} standard metrics...")
    loop = asyncio.get_running_loop()
    
    # Only use multiprocessing for interactive mode (no client)
    # Server mode should run sequentially/async to share the client
    use_mp = config.get("use_multiprocessing", True) and client is None
    
    if use_mp:
        # Multiprocessing mode (interactive)
        logger.info("Running Interactive Mode")
        pool_executor = concurrent.futures.ProcessPoolExecutor()
        
        for name, metric_class in metrics.items():
            logger.info(f"Running '{name}' for {len(examples)} examples")
            deps_for_stateless = {"repl_config": config.get("repl_config")}
            task_func = functools.partial(
                run_stateless_metric_in_process,
                metric_class,
                config.get(name, {}),
                deps_for_stateless,
                metric_name=name,
                global_config=config,
                column_names=column_names
            )
            
            tasks = [loop.run_in_executor(pool_executor, task_func, ex) for ex in examples]
            metric_results = await tqdm.gather(*tasks)
            results_df = save_and_log_results(df=results_df, metric=name, results=metric_results)
        
        pool_executor.shutdown()
    else:
        # Async mode (server)
        logger.info("Running Server Mode")
        for name, metric_class in metrics.items():
            logger.info(f"Running '{name}' for {len(examples)} examples")
            
            # Instantiate metric with client
            metric_instance = metric_class(config.get(name, {}), **shared_deps)
            
            # Run sequentially using run_check_async
            metric_results = []
            for ex in examples:
                args = example_to_metric_args(name, ex, config, column_names)
                res = await metric_instance.run_check_async(*args)
                metric_results.append(res)
            
            results_df = save_and_log_results(df=results_df, metric=name, results=metric_results)
    
    return results_df


async def run_batch_metrics(
    metrics: dict[str, BatchMetric],
    examples: list[dict[str, Any]],
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """Execute batch metrics on all examples at once.
    
    Batch metrics process all examples together, which is useful for
    metrics that benefit from batching (e.g., LLM-based metrics).
    
    Args:
        metrics: Dictionary of metric names to batch metric instances
        examples: List of examples to evaluate
        results_df: Results dataframe to update
        
    Returns:
        Updated results dataframe
    """
    if not metrics:
        return results_df
    
    logger.info(f"Running {len(metrics)} batch metrics...")
    
    for name, metric_instance in metrics.items():
        logger.info(f"Running '{name}' for {len(examples)} examples")
        metric_results = await metric_instance.run_batch_async(examples)
        results_df = save_and_log_results(df=results_df, metric=name, results=metric_results)
    
    return results_df

# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

async def evaluate(config_path: str):
    """Run the evaluation pipeline.
    
    This is the main entry point for LeanFlow evaluation. It orchestrates:
    1. Configuration loading and validation
    2. Data loading and preprocessing
    3. Metric loading and categorization
    4. Metric execution (stateless and batch)
    5. Results saving
    
    Args:
        config_path: Path to YAML configuration file
        
    The evaluation supports two execution modes:
    - Interactive mode: Uses REPL with multiprocessing
    - Server mode: Uses Client with async execution
    
    Results are saved as JSON to the output directory specified in the config.
    """
    # Load configuration
    try:
        conf = OmegaConf.load(config_path)
    except FileNotFoundError:
        logger.critical(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed to parse configuration file '{config_path}': {e}")
        sys.exit(1)

    # Setup environment
    setup_environment(conf)

    # Load column names from config with defaults
    column_names = {**DEFAULT_COLUMN_NAMES, **conf.get("column_names", {})}
    logger.debug(f"Column name mapping: {column_names}")

    # Load and preprocess data
    try:
        examples = load_data(conf.data_path)
        examples = preprocess_data(examples, conf, column_names)
        logger.info(f"Loaded {len(examples)} examples from {conf.data_path}")
    except FileNotFoundError as e:
        logger.critical(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Data loading failed: {e}")
        sys.exit(1)
    
    # Prepare output directory and results dataframe
    output_dir = Path(conf.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ds_path = output_dir / Path(conf.data_path).name.replace(".json", "")
    results_df = pd.DataFrame(examples)

    # Setup runner (client or repl_config)
    shared_deps, client = setup_runner(conf)

    # Load metrics
    try:
        stateless_metrics, batch_metrics = load_metrics(conf, results_df, shared_deps)
        logger.info(f"Loaded {len(stateless_metrics)} stateless and {len(batch_metrics)} batch metrics")
    except Exception as e:
        logger.critical(f"Failed to load metrics: {e}")
        sys.exit(1)

    # Run stateless metrics
    try:
        results_df = await run_stateless_metrics(
            stateless_metrics, examples, conf, shared_deps, client, results_df, column_names=column_names
        )
    except Exception as e:
        logger.error(f"Error running stateless metrics: {e}")
        raise

    # Run batch metrics
    try:
        results_df = await run_batch_metrics(batch_metrics, examples, results_df)
    except Exception as e:
        logger.error(f"Error running batch metrics: {e}")
        raise

    # Save results
    try:
        output_file = f"{output_ds_path}.json"
        results_df.to_json(output_file, orient="records", indent=2)
        logger.debug(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

    logger.success("Evaluation script finished.")


def main():
    """Main entry point for leanflow-eval command.
    
    Parses command-line arguments and runs the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate autoformalization models using the LeanFlow framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with config file
  python -m leanflow.evaluate_cli --config config.yaml
  
  # Using the leanflow-eval command (after installation)
  leanflow-eval --config config.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    asyncio.run(evaluate(args.config))


if __name__ == "__main__":
    main()