"""Logging configuration for LLM Compare."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    session_id: Optional[str] = None
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        session_id: Optional session ID to include in log file name
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Get root logger for llm_compare
    root_logger = logging.getLogger("llm_compare")
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        if session_id:
            log_file = log_file.parent / f"{log_file.stem}_{session_id}{log_file.suffix}"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"llm_compare.{name}")


class SessionLogger:
    """Logger that writes to a session-specific log file."""

    def __init__(self, session_dir: Path, session_id: str):
        """
        Initialize session logger.

        Args:
            session_dir: Directory for session files
            session_id: Unique session identifier
        """
        self.session_dir = session_dir
        self.session_id = session_id
        self.log_file = session_dir / "session.log"
        self.logger = get_logger(f"session.{session_id}")

        # Add file handler for session log
        session_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def phase_start(self, phase_name: str) -> None:
        """Log the start of an evaluation phase."""
        self.logger.info(f"=== Starting Phase: {phase_name} ===")

    def phase_end(self, phase_name: str) -> None:
        """Log the end of an evaluation phase."""
        self.logger.info(f"=== Completed Phase: {phase_name} ===")

    def provider_call(self, provider: str, action: str) -> None:
        """Log a provider API call."""
        self.logger.debug(f"[{provider}] {action}")

    def evaluation_result(self, evaluator: str, evaluated: str, score: float) -> None:
        """Log an evaluation result."""
        self.logger.info(f"Evaluation: {evaluator} -> {evaluated}: {score:.2f}")
