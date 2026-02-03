from typing import Union, Optional

from .dataclasses import Environment

def get_env_id(env: Optional[Union[Environment, int]]) -> Optional[int]:
    """Extracts an environment ID from various input types.

    Args:
        env (Optional[Union[Environment, int]]): An Environment object, an integer ID, or None.

    Returns:
        Optional[int]: The integer environment ID, or None if the input is None.
    """
    if isinstance(env, Environment):
        return env.env
    
    if isinstance(env, int) and env>=0:
        return env
        
    return None

class NoOpAsyncContextManager:
    """A no-op async context manager for reusable resources that shouldn't be closed."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass