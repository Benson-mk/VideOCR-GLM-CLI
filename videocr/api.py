import sys

from . import utils
from .exceptions import OllamaConnectionError, OllamaModelError, VideoProcessingError
from .video import Video


def save_subtitles_to_file(
        video_path: str, file_path='subtitle.srt', lang='en', time_start='0:00', time_end='',
        sim_threshold=80, max_merge_gap_sec=0.1, use_fullframe=False,
        brightness_threshold=None, ssim_threshold=92, subtitle_position="center",
        frames_to_skip=1, crop_zones=None, ocr_image_max_width=960,
        post_processing=False, min_subtitle_duration_sec=0.2,
        ollama_host='localhost', ollama_port=11434, ollama_model='glm-ocr:latest',
        ollama_timeout=300) -> None:
    """
    Extract subtitles from video and save to SRT file.
    
    Args:
        video_path: Path to the video file
        file_path: Output SRT file path
        lang: OCR language code
        time_start: Start time for extraction (MM:SS or HH:MM:SS)
        time_end: End time for extraction (MM:SS or HH:MM:SS)
        sim_threshold: Similarity threshold for merging subtitles (0-100)
        max_merge_gap_sec: Maximum time gap in seconds to merge similar subtitles
        use_fullframe: Use full frame for OCR instead of crop zones
        brightness_threshold: Brightness threshold for image preprocessing
        ssim_threshold: SSIM similarity threshold for frame skipping (0-100)
        subtitle_position: Subtitle position alignment (center, left, right, any)
        frames_to_skip: Number of frames to skip between OCR operations
        crop_zones: List of crop zone dictionaries
        ocr_image_max_width: Maximum image width for OCR
        post_processing: Enable text post-processing
        min_subtitle_duration_sec: Minimum subtitle duration in seconds
        ollama_host: Ollama server host
        ollama_port: Ollama server port
        ollama_model: Ollama model name
        ollama_timeout: Ollama API timeout in seconds
    
    Raises:
        OllamaConnectionError: If connection to Ollama fails
        OllamaModelError: If the specified model is not available
        VideoProcessingError: If video processing fails
    """
    if crop_zones is None:
        crop_zones = []

    # Verify Ollama connection
    try:
        utils.check_ollama_connection(ollama_host, ollama_port, ollama_model, ollama_timeout)
    except (OllamaConnectionError, OllamaModelError) as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)

    v = Video(video_path, ollama_host, ollama_port, ollama_model, ollama_timeout, time_end)
    try:
        v.run_ocr(
            lang, time_start, time_end, use_fullframe, brightness_threshold,
            ssim_threshold, subtitle_position, frames_to_skip, crop_zones,
            ocr_image_max_width
        )
    except (ValueError, Exception) as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
    
    subtitles = v.get_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec)

    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(subtitles)
