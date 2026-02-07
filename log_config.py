"""
Log Configuration Module
Provides setup for logging productivity monitoring events.
"""

import logging
import os
from datetime import datetime


def setup_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file output.
    
    Args:
        log_file: Name of the log file
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Full path to log file
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    # Console handler (optional - can be removed if you only want file logging)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in console
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_session_start(logger: logging.Logger):
    """Log the start of a monitoring session."""
    logger.info("=" * 60)
    logger.info("PRODUCTIVITY MONITORING SESSION STARTED")
    logger.info(f"Session started at: {datetime.now()}")
    logger.info("=" * 60)


def log_session_end(logger: logging.Logger):
    """Log the end of a monitoring session."""
    logger.info("=" * 60)
    logger.info("PRODUCTIVITY MONITORING SESSION ENDED")
    logger.info(f"Session ended at: {datetime.now()}")
    logger.info("=" * 60)
