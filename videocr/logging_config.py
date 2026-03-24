"""
Logging configuration for VideOCR.

This module sets up centralized logging configuration for the application,
replacing print statements with proper logging.
"""

import logging
import sys
from pathlib import Path

# OCR log file paths - initialized to None, will be set by utils module
# These store the paths to OCR-specific log files for debugging purposes
_OCR_RESPONSE_LOG_PATH = None  # Path to log file for raw OCR API responses
_OCR_PAYLOAD_LOG_PATH = None   # Path to log file for OCR API request payloads


def enable_ocr_logging(response: bool = True, payload: bool = True, log_image: bool = False) -> None:
    """
    Enable OCR logging for debugging purposes.
    
    This function controls the creation of OCR-specific log files:
    - ocr_raw_responses.log: Logs raw OCR API responses
    - ocr_api_payloads.log: Logs OCR API request payloads
    
    Args:
        response: Enable logging of OCR API responses (default: True)
        payload: Enable logging of OCR API request payloads (default: True)
        log_image: Include base64 image data in payload logs (default: False - can be very large)
    
    Note:
        Log files are created in:
        - Windows: %LOCALAPPDATA%\\VideOCR\\
        - Linux/Mac: ~/.config/VideOCR/
    """
    from . import utils
    utils.enable_ocr_logging(response=response, payload=payload, log_image=log_image)


def disable_ocr_logging() -> None:
    """
    Disable all OCR logging to prevent log file creation.
    
    This stops the creation of both ocr_raw_responses.log and ocr_api_payloads.log files.
    Useful for production use or when debugging is not needed.
    """
    from . import utils
    utils.disable_ocr_logging()


def is_ocr_logging_enabled() -> bool:
    """
    Check if OCR logging is currently enabled.
    
    Returns:
        True if OCR logging is enabled, False otherwise
    """
    from . import utils
    return utils.is_ocr_logging_enabled()


def setup_logging(
    log_level: int = logging.INFO,
    log_to_file: bool = False,
    log_file: str | None = None,
    log_dir: str | None = None
) -> None:
    """
    Set up logging configuration for VideOCR.
    
    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to file in addition to console
        log_file: Specific log file path (optional)
        log_dir: Directory for log files (used if log_file not specified)
    """
    # Create logger
    logger = logging.getLogger('videocr')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        if log_file:
            log_path = Path(log_file)
        elif log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_path = log_dir_path / 'videocr.log'
        else:
            # Default to user's config directory
            import platform
            if platform.system() == "Windows":
                log_dir_path = Path.home() / "AppData" / "Local" / "VideOCR"
            else:
                log_dir_path = Path.home() / ".config" / "VideOCR"
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_path = log_dir_path / 'videocr.log'
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Set up module-level loggers
    _setup_module_loggers(log_level)


def _setup_module_loggers(log_level: int) -> None:
    """Set up loggers for specific modules."""
    modules = [
        'videocr.video',
        'videocr.utils',
        'videocr.api',
        'videocr.models'
    ]
    
    for module_name in modules:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)