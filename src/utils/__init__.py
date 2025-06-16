"""
Utilities Module for Clickbait Detection
=======================================

Common utilities used across the project including:
- Logging configuration
- Configuration management
- GPU utilities
- File I/O helpers
"""

from .logging_utils import setup_logger, configure_logging
from .config import ConfigManager
from .gpu_utils import GPUManager
from .file_utils import FileManager

__all__ = [
    "setup_logger",
    "configure_logging", 
    "ConfigManager",
    "GPUManager",
    "FileManager"
] 