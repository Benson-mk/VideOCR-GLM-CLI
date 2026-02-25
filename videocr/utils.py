import datetime
import logging
import logging.handlers
import os
import platform
import sys
import json
import base64

import cv2
import requests

from .exceptions import OllamaConnectionError, OllamaModelError, OCRApiError, OCRApiTimeoutError
from .lang_dictionaries import (
    ARABIC_LANGS, CYRILLIC_LANGS, DEVANAGARI_LANGS,
    ESLAV_LANGS, LATIN_LANGS,
)


# Log Configuration
_OCR_RESPONSE_LOG = False
_OCR_PAYLOAD_LOG = False
_OCR_PAYLOAD_LOG_IMAGE = False

# Maximum log file size: 10MB
_MAX_LOG_BYTES = 10 * 1024 * 1024
# Number of backup log files to keep
_BACKUP_COUNT = 5

_OCR_RESPONSE_LOGGER = None
_OCR_PAYLOAD_LOGGER = None
_OCR_RESPONSE_LOG_PATH = None
_OCR_PAYLOAD_LOG_PATH = None


def _get_log_dir() -> str:
    """Get or create the log directory."""
    if platform.system() == "Windows":
        log_dir = os.path.join(os.getenv('LOCALAPPDATA'), "VideOCR")
    else:
        log_dir = os.path.join(os.path.expanduser('~'), ".config", "VideOCR")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _get_ocr_response_logger() -> logging.Logger:
    """Get or create the OCR response logger with rotation."""
    global _OCR_RESPONSE_LOGGER
    if _OCR_RESPONSE_LOGGER is None:
        logger = logging.getLogger('videocr.ocr_response')
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            log_dir = _get_log_dir()
            log_path = os.path.join(log_dir, "ocr_raw_responses.log")
            
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=_MAX_LOG_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding='utf-8'
            )
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        
        _OCR_RESPONSE_LOGGER = logger
    return _OCR_RESPONSE_LOGGER


def _get_ocr_payload_logger() -> logging.Logger:
    """Get or create the OCR payload logger with rotation."""
    global _OCR_PAYLOAD_LOGGER
    if _OCR_PAYLOAD_LOGGER is None:
        logger = logging.getLogger('videocr.ocr_payload')
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            log_dir = _get_log_dir()
            log_path = os.path.join(log_dir, "ocr_api_payloads.log")
            
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=_MAX_LOG_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding='utf-8'
            )
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        
        _OCR_PAYLOAD_LOGGER = logger
    return _OCR_PAYLOAD_LOGGER


def _get_ocr_response_log_path() -> str:
    """Get or create the OCR response log file path."""
    global _OCR_RESPONSE_LOG_PATH
    if _OCR_RESPONSE_LOG_PATH is None:
        if platform.system() == "Windows":
            log_dir = os.path.join(os.getenv('LOCALAPPDATA'), "VideOCR")
        else:
            log_dir = os.path.join(os.path.expanduser('~'), ".config", "VideOCR")
        os.makedirs(log_dir, exist_ok=True)
        _OCR_RESPONSE_LOG_PATH = os.path.join(log_dir, "ocr_raw_responses.log")
    return _OCR_RESPONSE_LOG_PATH


def _get_ocr_payload_log_path() -> str:
    """Get or create the OCR API payload log file path."""
    global _OCR_PAYLOAD_LOG_PATH
    if _OCR_PAYLOAD_LOG_PATH is None:
        if platform.system() == "Windows":
            log_dir = os.path.join(os.getenv('LOCALAPPDATA'), "VideOCR")
        else:
            log_dir = os.path.join(os.path.expanduser('~'), ".config", "VideOCR")
        os.makedirs(log_dir, exist_ok=True)
        _OCR_PAYLOAD_LOG_PATH = os.path.join(log_dir, "ocr_api_payloads.log")
    return _OCR_PAYLOAD_LOG_PATH


def _log_ocr_response(frame_index: int, lang: str, raw_text: str, 
                      is_valid: bool, zone_idx: int = 0) -> None:
    """Log raw VLM response to file for debugging."""
    log_path = _get_ocr_response_log_path()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    log_entry = {
        "timestamp": timestamp,
        "frame_index": frame_index,
        "zone_index": zone_idx,
        "language": lang,
        "is_valid": is_valid,
        "raw_response": raw_text,
        "response_length": len(raw_text) if raw_text else 0
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def _log_api_payload(frame_index: int, lang: str, zone_idx: int, 
                     payload: dict, log_image: bool = _OCR_PAYLOAD_LOG_IMAGE) -> None:
    """
    Log API request payload to file for debugging.
    
    Args:
        frame_index: Current frame index
        lang: Language code
        zone_idx: Zone index
        payload: Full API payload dict
        log_image: Whether to include base64 image in log (default: False - too large)
    """
    log_path = _get_ocr_payload_log_path()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Create a copy of payload, optionally excluding large image data
    log_payload = payload.copy()
    if not log_image and 'images' in log_payload:
        # Store image dimensions instead of full base64
        img_data = log_payload['images'][0] if log_payload['images'] else None
        if img_data:
            log_payload['images'] = [{
                'type': 'base64',
                'length': len(img_data),
                'preview': img_data[:50] + '...' if len(img_data) > 50 else img_data
            }]
    
    log_entry = {
        "timestamp": timestamp,
        "frame_index": frame_index,
        "zone_index": zone_idx,
        "language": lang,
        "api_endpoint": f"http://{{host}}:{{port}}/api/generate",
        "payload": log_payload
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n")


def get_frame_index(time_str: str, fps: float) -> int:
    t = time_str.split(':')
    t = list(map(float, t))
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(f'Time data "{time_str}" does not match format "%H:%M:%S"')
    total_seconds = td.total_seconds()
    return max(0, int(total_seconds * fps))


def get_ms_from_time_str(time_str: str) -> float:
    t = time_str.split(':')
    t = list(map(float, t))
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(f'Time data "{time_str}" does not match format "%H:%M:%S"')
    return td.total_seconds() * 1000


def get_frame_index_from_ms(frame_timestamps: dict[int, float], target_ms: float) -> int:
    return min(frame_timestamps.items(), key=lambda item: abs(item[1] - target_ms))[0]


def get_srt_timestamp(frame_index: int, fps: float, offset_ms: float = 0.0) -> str:
    td = datetime.timedelta(milliseconds=(frame_index / fps * 1000 + offset_ms))
    ms = td.microseconds // 1000
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def get_srt_timestamp_from_ms(ms: float) -> str:
    td = datetime.timedelta(milliseconds=ms)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def extract_non_chinese_segments(text) -> list[tuple[str, str]]:
    segments = []
    current_segment = ''

    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'

    for char in text:
        if is_chinese(char):
            if current_segment:
                segments.append(('non_chinese', current_segment))
                current_segment = ''
            segments.append(('chinese', char))
        else:
            current_segment += char

    if current_segment:
        segments.append(('non_chinese', current_segment))

    return segments


def check_ollama_connection(host: str, port: int, model: str, timeout: int) -> None:
    """
    Verify Ollama is running and model is available.
    
    Args:
        host: Ollama server host address
        port: Ollama server port
        model: Model name to check
        timeout: Connection timeout in seconds
    
    Raises:
        OllamaConnectionError: If connection to Ollama fails
        OllamaModelError: If the specified model is not available
    """
    url = f"http://{host}:{port}/api/tags"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if model not in model_names:
            raise OllamaModelError(
                f"Model '{model}' not found in Ollama.\n"
                f"Available models: {', '.join(model_names)}\n"
                f"Run: ollama pull {model}"
            )
        print(f"✓ Connected to Ollama at {host}:{port}")
        print(f"✓ Model '{model}' is available")
    except requests.exceptions.ConnectionError:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {host}:{port}\n"
            f"Make sure Ollama is running: ollama serve"
        )
    except OllamaModelError:
        raise
    except Exception as e:
        raise OllamaConnectionError(f"Error checking Ollama connection: {e}")


def encode_image_to_base64(image: cv2.Mat) -> str:
    """Convert OpenCV image to base64 string for API."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def is_blank_subtitle(text: str) -> bool:
    """
    Detect if GLM-OCR output is actually a blank/no-subtitle frame.
    GLM-OCR (VLM) may hallucinate text when none exists.
    """
    if not text or not text.strip():
        return True
    
    text_lower = text.lower().strip()
    
    # Common VLM responses when no text exists
    blank_patterns = [
        'no text', 'no text found', 'no text visible', 'no text detected',
        'no subtitles', 'no caption', 'no visible text',
        'empty', 'blank', 'cannot see', 'unable to find',
        'i cannot', 'i do not', "i don't", 'there is no',
        'there are no', 'the image', 'this image', 'shows a',
        'shows the', 'dark bar', 'black bar', 'bottom of',
        'top of', 'scene', 'frame', 'video', 'background',
        'appears to be', 'no visible', "i don't see",
        'i do not see', 'no words', 'nothing written',
        'complete extracted text',
    ]
    
    for pattern in blank_patterns:
        if pattern in text_lower:
            return True
    
    # Too short (likely noise) - less than 2 characters
    if len(text_lower) < 2:
        return True
    
    # All punctuation/no alphanumeric
    if not any(c.isalnum() for c in text):
        return True
    
    return False


def call_glm_ocr(image: cv2.Mat, host: str, port: int, model: str,
                 timeout: int, lang: str = 'en', frame_index: int = 0,
                 zone_idx: int = 0, log_payload: bool = _OCR_PAYLOAD_LOG,
                 log_response: bool = _OCR_RESPONSE_LOG) -> tuple[str, bool]:
    """
    Call GLM-OCR via Ollama API.
    
    Args:
        image: OpenCV image frame
        host: Ollama host
        port: Ollama port
        model: Ollama model name
        timeout: Request timeout in seconds
        lang: Language code for OCR
        frame_index: Current frame index (for logging)
        zone_idx: Zone index (for logging)
        log_payload: Whether to log API request payload (default: False)
        log_response: Whether to log raw VLM response (default: False)
    
    Returns:
        tuple: (extracted_text, is_valid)
               - is_valid=False means no subtitle detected (blank frame)
    """
    url = f"http://{host}:{port}/api/generate"
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(image)
    
    # Build prompt based on language
    lang_prompts = {
        'ch': 'Extract all Chinese text from this image. If no text is found, output exactly: [NO_TEXT]. Output only the text or [NO_TEXT], nothing else.',
        'chinese_cht': 'Extract all Traditional Chinese text from this image. If no text is found, output exactly: [NO_TEXT]. Output only the text or [NO_TEXT], nothing else.',
        'en': 'Extract all English text from this image. If no text is found, output exactly: [NO_TEXT]. Output only the text or [NO_TEXT], nothing else.',
        'ja': 'Extract all Japanese text from this image. If no text is found, output exactly: [NO_TEXT]. Output only the text or [NO_TEXT], nothing else.',
        'ko': 'Extract all Korean text from this image. If no text is found, output exactly: [NO_TEXT]. Output only the text or [NO_TEXT], nothing else.',
    }
    prompt = lang_prompts.get(lang, f'Extract all text in {lang} from this image. If no text is found, output exactly: [NO_TEXT]. Output only the text or [NO_TEXT], nothing else.')
    
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.05,
            "top_p": 0.9,
        }
    }
    
    # Log API payload before sending request
    if log_payload:
        _log_api_payload(frame_index, lang, zone_idx, payload, log_image=_OCR_PAYLOAD_LOG_IMAGE)
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        text = result.get('response', '').strip()
        
        # Log raw VLM response
        if log_response:
            _log_ocr_response(frame_index, lang, text, text != '', zone_idx)
        
        # Check for explicit no-text marker
        if text.upper() == '[NO_TEXT]':
            return '', False
        
        # Check for blank/hallucination patterns
        if is_blank_subtitle(text):
            return '', False
        
        return text, True
        
    except requests.exceptions.Timeout:
        if log_response:
            _log_ocr_response(frame_index, lang, f"TIMEOUT after {timeout}s", False, zone_idx)
        raise TimeoutError(f"Ollama API timeout after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        if log_response:
            _log_ocr_response(frame_index, lang, f"ERROR: {str(e)}", False, zone_idx)
        raise RuntimeError(f"Ollama API error: {e}")


def log_error(message: str, log_name: str = "error_log.txt") -> str:
    if platform.system() == "Windows":
        log_dir = os.path.join(os.getenv('LOCALAPPDATA'), "VideOCR")
    else:
        log_dir = os.path.join(os.path.expanduser('~'), ".config", "VideOCR")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
    return log_path


def is_process_running(pid: int) -> bool:
    try:
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            return str(pid) in result.stdout
        else:
            if os.path.exists(f"/proc/{pid}"):
                return True
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return False