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
import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

_LOGGER_IS_CONFIGURED = False

def setup_logger(log_dir: Optional[Union[str, Path]] = None, log_level : Optional[str] = None, debug: bool = False):
    """Configure the logger.

    Args:
        log_dir (Optional[Union[str, Path]]): An optional, explicit directory to save log files.
            If not provided, it falls back to the LEANFLOW_LOG_DIR environment variable,
            and then to the default `~/.leanflow/logs`.
        log_level (Optional[str]): The logging level (e.g., "INFO", "DEBUG"). Defaults to "INFO".
        debug (Optional[bool]): If True, enables file logging even if log_dir is not provided.
    """
    global _LOGGER_IS_CONFIGURED
    if _LOGGER_IS_CONFIGURED:
        return

    logger.remove()

    LEVEL_CONFIG = {
        "TRACE":   {"icon": "🔍"},
        "DEBUG":   {"icon": "🐛"},
        "INFO":    {"icon": "📢"},
        "SUCCESS": {"icon": "✅"},
        "WARNING": {"icon": "⚠️"},
        "ERROR":   {"icon": "❌"},
        "CRITICAL":{"icon": "🔥"},
    }

    def formatter(record):
        """Custom formatter to add icons and handle exceptions cleanly."""

        if record["level"].name in LEVEL_CONFIG:
            record["extra"]["icon"] = LEVEL_CONFIG[record["level"].name]["icon"]
        else:
            record["extra"]["icon"] = ""

        if record["exception"]:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "{level.name: <5} | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>\n"
                "{exception}"
            )
        return (
            "<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue> | "
            "<level>{extra[icon]}</level> | "
            "<cyan><level>{message}</level></cyan>\n"
        )

    if log_level is None:
        log_level = "INFO"
    logger.add(sys.stderr, level=str(log_level).upper(), format=formatter, colorize=True)

    # determine log directory based on priority and configure file logger
    save_to_file = debug
    log_path = None
    if log_dir:
        log_path = Path(log_dir)
        save_to_file = True
    elif os.environ.get("LEANFLOW_LOG_DIR"):
        log_path = Path(os.environ.get("LEANFLOW_LOG_DIR"))
        save_to_file = True
    else:
        log_path = Path.home() / ".leanflow" / "logs"

    if save_to_file:
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / "leanflow_{time}.log"

        logger.debug(f"File logging configured at: {log_path}")

        logger.add(
            log_file_path,
            level="TRACE",
            rotation="500 MB",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <5} | {function}:{line} - {message}"
        )
    
    _LOGGER_IS_CONFIGURED = True

# Run a default setup in case no entry point explicitly calls setup_logger
if not _LOGGER_IS_CONFIGURED:
    setup_logger()