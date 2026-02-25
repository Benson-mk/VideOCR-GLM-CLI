"""
Custom exception classes for VideOCR.

This module defines specific exceptions for better error handling
and debugging throughout the application.
"""


class VideOCRError(Exception):
    """Base exception for all VideOCR errors."""
    pass


class OllamaConnectionError(VideOCRError):
    """Raised when connection to Ollama server fails."""
    pass


class OllamaModelError(VideOCRError):
    """Raised when the specified Ollama model is not available."""
    pass


class VideoProcessingError(VideOCRError):
    """Raised when video processing fails."""
    pass


class VideoValidationError(VideOCRError):
    """Raised when video validation fails (e.g., invalid path, format)."""
    pass


class OCRApiError(VideOCRError):
    """Raised when OCR API call fails."""
    pass


class OCRApiTimeoutError(OCRApiError):
    """Raised when OCR API call times out."""
    pass


class CropZoneError(VideOCRError):
    """Raised when crop zone parameters are invalid."""
    pass


class TimeRangeError(VideOCRError):
    """Raised when time range parameters are invalid."""
    pass