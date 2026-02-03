from abc import ABC, abstractmethod
from typing import Any, Optional
import asyncio

from ..utils import setup_logger, NoOpAsyncContextManager
from ..client import Client
from ..errors import LeanValueError

class Metric(ABC):
    """Abstract base class for stateless metrics.

    The `compute` method is synchronous, making it compatible with multiprocessing.
    """

    def __init__(
        self,
        metric_config: dict[str, Any] = {},
        repl_config: dict[str, Any] = {},
        client: Optional[Client] = None,
        **shared_dependencies
    ):
        """Initializes the metric.

        Args:
            metric_config (dict[str, Any]): Configuration dictionary for the metric.
            repl_config (dict[str, Any]): Configuration for REPL/Client (e.g. {'lean_version': ...}).
            client (Optional[Client]): Existing client instance to share.
            **shared_dependencies (Any): Other shared dependencies.
        """
        self.metric_config = metric_config
        self.repl_config = repl_config
        self.client = client
        for key, value in shared_dependencies.items():
            setattr(self, key, value)
        
        log_dir = (
            self.metric_config.get("log_dir") or 
            getattr(self, "log_dir", None) or 
            getattr(self.__class__, "log_dir", None)
        )
        
        log_level = (
            self.metric_config.get("log_level") or 
            getattr(self, "log_level", None) or 
            getattr(self.__class__, "log_level", None)
        )
        
        setup_logger(log_dir, log_level)
    
    @property
    def name(self) -> str:
        """Returns the snake_case name of the metric class.

        Returns:
            (str): The name of the metric.
        """
        import re
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()

    def compute(self, *args, **kwargs) -> Any:
        """Synchronously computes the metric for a single example.

        This method is the entrypoint for a multiprocessing worker. It calls the REPL
        asynchronously and returns the result.

        Args:
            *args (Any): Positional arguments for the metric check.
            **kwargs (Any): Keyword arguments for the metric check.

        Returns:
            (Any): The result of the metric computation.
        """
        return asyncio.run(self.run_check_async(*args, **kwargs))
    
    @abstractmethod
    async def run_check_async(self, *args, **kwargs) -> Any:
        """Asynchronously computes the metric for a single example.

        Args:
            *args (Any): Positional arguments for the metric check.
            **kwargs (Any): Keyword arguments for the metric check.

        Returns:
            (Any): The result of the metric computation.
        """
        pass

    def get_runner(self) -> Any:
        """Returns the appropriate LeanRunner and context manager.
        
        Returns:
            (tuple[runner, context]): The runner and an appropriate context manager.
                - For reusable clients: (client, NoOpAsyncContextManager())
                - For fresh instances: (instance, instance)
        """
        from ..repl import REPL

        # Use explicit client if provided (reusable, managed externally)
        if self.client:
            return self.client, NoOpAsyncContextManager()
        
        # Check repl_config (create fresh instance)
        repl_config = self.repl_config
        if repl_config:
            if "base_url" in repl_config:
                client = Client(**repl_config)
                return client, client
            
            runner = REPL(**repl_config)
            return runner, runner
            
        raise LeanValueError(f"{self.__class__.__name__} requires either 'client' or 'repl_config'.")

class BatchMetric(Metric):
    """Base class for metrics that can be computed on a batch of examples."""

    def compute_batch(self, examples: list[Any]) -> Any:
        """Synchronously computes the metric for a batch of examples.

        Args:
            examples (list[Any]): A list of examples to process.

        Returns:
            (Any): The result of the batch computation.

        Raises:
            LeanValueError: If `examples` is not a list.
        """
        if not isinstance(examples, list):
            raise LeanValueError(f"Expected a list of examples, got {type(examples)}")
        return asyncio.run(self.run_batch_async(examples))
    
    @abstractmethod
    async def run_batch_async(self, examples: list[Any]) -> Any:
        """Asynchronously computes the metric for a batch of examples.

        Args:
            examples (list[Any]): List of examples to process.

        Returns:
            (Any): The result of the metric computation.
        """
        pass

    async def run_check_async(self, *args, **kwargs) -> Any:
        return self.run_batch_async(*args, **kwargs)