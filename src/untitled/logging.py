# Copyright (c) 2025 untitled

import logging
import os
import sys

# Define color codes for different log levels
colors = {
    # Cyan
    "DEBUG": "\033[0;36m",
    # Green
    "INFO": "\033[0;32m",
    # Yellow
    "WARNING": "\033[0;33m",
    # Red
    "ERROR": "\033[0;31m",
    # Bold Red
    "CRITICAL": "\033[1;31m",
    # Reset color
    "RESET": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels."""

    def format(self, record):
        """Format the log record with colored level names."""
        levelname = record.levelname
        if levelname in colors:
            record.levelname = f"{colors[levelname]}{levelname}{colors['RESET']}"
        return super().format(record)


def get_logger(name: str, level: int) -> logging.Logger:
    """Get a logger instance with colored output.

    Args:
        name (str): Name of the logger, typically __name__ or module path
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING,
                     logging.ERROR, logging.CRITICAL). Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(level)

        # Configure console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Configure formatter
        formatter = ColorFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Configure handler to logger
        logger.addHandler(console_handler)

    return logger


def get_info_logger(name: str) -> logging.Logger:
    """Get a logger instance with INFO level."""
    return get_logger(name, logging.INFO)


def get_debug_logger(name: str) -> logging.Logger:
    """Get a logger instance with DEBUG level."""
    return get_logger(name, logging.DEBUG)


def get_warning_logger(name: str) -> logging.Logger:
    """Get a logger instance with WARNING level."""
    return get_logger(name, logging.WARNING)


def get_error_logger(name: str) -> logging.Logger:
    """Get a logger instance with ERROR level."""
    return get_logger(name, logging.ERROR)


def get_critical_logger(name: str) -> logging.Logger:
    """Get a logger instance with CRITICAL level."""
    return get_logger(name, logging.CRITICAL)


def get_logger_from_env(name: str) -> logging.Logger:
    """Get a logger instance with level configured from LOG_LEVEL environment variable."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level, logging.INFO)
    return get_logger(name, level)
