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

import os
import psutil

def get_available_cpus(default: int = 1) -> int:
    """Get the number of CPUs available to the current process.

    Args:
        default: Value to return if no method succeeds. Defaults to 1.

    Returns:
        int: Number of available CPUs, or default if detection fails.
    """
    
    try:
        count = os.process_cpu_count()
        if count is not None and count > 0:
            return count
    except AttributeError:
        pass  # Python < 3.13

    try:
        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            affinity = p.cpu_affinity()
            if affinity:
                return len(affinity)
    except Exception:
        pass

    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except Exception:
        pass

    try:
        count = os.cpu_count()
        if count is not None and count > 0:
            return count
    except Exception:
        pass

    return default
