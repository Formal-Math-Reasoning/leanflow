"""System utility functions for LeanFlow."""

import os


def get_available_cpus(default: int = 1) -> int:
    """Get the number of CPUs available to the current process.

    This function is designed to work correctly in various environments including:
    - SLURM managed clusters (respects CPU allocation)
    - Docker/containerized environments (respects cgroup limits)
    - Standard systems

    The function tries multiple methods in order of preference:
    1. os.process_cpu_count() (Python 3.13+) - respects cgroups and process affinity
    2. os.sched_getaffinity() (Unix) - gets CPUs the process can run on
    3. os.cpu_count() - total system CPUs (may be incorrect in containers/SLURM)

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
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except (AttributeError, OSError):
        pass  # Not available on all platforms (e.g., macOS, Windows)

    # Method 3: Fallback to cpu_count (may be incorrect in SLURM/containers)
    try:
        count = os.cpu_count()
        if count is not None and count > 0:
            return count
    except Exception:
        pass

    return default
