"""Centralized logging configuration for lemma."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def get_log_dir() -> Path:
    """Get the lemma log directory."""
    log_dir = Path.home() / ".lemma"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (usually __name__ from calling module)
        level: Optional log level override (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    # Determine log level
    log_level: int
    if level is None:
        log_level = logging.INFO
    elif isinstance(level, str):
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = level

    logger.setLevel(log_level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    # File handler - detailed logs to ~/.lemma/app.log
    try:
        log_file = get_log_dir() / "app.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # If we can't create log file, just log to console
        print(f"Warning: Could not create log file: {e}", file=sys.stderr)

    # Console handler - simple logs to stderr (only WARNING and above)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Log an exception with context.

    Args:
        logger: Logger instance
        message: Context message
        exc: Exception to log
    """
    logger.error(f"{message}: {type(exc).__name__}: {str(exc)}", exc_info=True)
