"""System utility functions for LeanFlow."""

import os


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
