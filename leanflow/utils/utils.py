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
import threading
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

class _AsyncRunner:
    """Runs async coroutines from synchronous contexts, including Jupyter notebooks. Uses a background thread with its own event loop so that `run()` works even when called from within an already-running loop (e.g. Jupyter/IPython)."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        """Submit a coroutine and block until it completes."""
        if self._loop is None or self._loop.is_closed():
            self._start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        """Stop the background loop and join the thread."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None


class NoOpAsyncContextManager:
    """A no-op async context manager for reusable resources that shouldn't be closed."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass