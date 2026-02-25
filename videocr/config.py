"""
Configuration management for VideOCR.

This module provides a centralized configuration class to manage
logging settings and other global configuration options.
"""


class OCRConfig:
    """
    Configuration class for OCR operations and logging.
    
    This class centralizes configuration options that were previously
    managed as global variables, making the code more testable and
    maintainable.
    """
    
    def __init__(
        self,
        log_response: bool = False,
        log_payload: bool = False,
        log_image: bool = False,
        max_log_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ):
        """
        Initialize OCR configuration.
        
        Args:
            log_response: Whether to log OCR API responses
            log_payload: Whether to log OCR API request payloads
            log_image: Whether to include base64 image data in payload logs
            max_log_bytes: Maximum size of log files before rotation (default: 10MB)
            backup_count: Number of backup log files to keep (default: 5)
        """
        self.log_response = log_response
        self.log_payload = log_payload
        self.log_image = log_image
        self.max_log_bytes = max_log_bytes
        self.backup_count = backup_count
    
    def enable_logging(self, response: bool = True, payload: bool = True) -> None:
        """
        Enable logging options.
        
        Args:
            response: Enable response logging
            payload: Enable payload logging
        """
        self.log_response = response
        self.log_payload = payload
    
    def disable_logging(self) -> None:
        """Disable all logging options."""
        self.log_response = False
        self.log_payload = False
        self.log_image = False
    
    def __repr__(self) -> str:
        return (
            f"OCRConfig(log_response={self.log_response}, "
            f"log_payload={self.log_payload}, "
            f"log_image={self.log_image})"
        )


# Default global configuration instance
_default_config = OCRConfig()


def get_default_config() -> OCRConfig:
    """
    Get the default global configuration instance.
    
    Returns:
        The default OCRConfig instance
    """
    return _default_config


def set_default_config(config: OCRConfig) -> None:
    """
    Set the default global configuration instance.
    
    Args:
        config: New configuration instance to use as default
    """
    global _default_config
    _default_config = config