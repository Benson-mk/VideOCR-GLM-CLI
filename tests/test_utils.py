"""
Unit tests for videocr.utils module.
"""

import pytest
from videocr.utils import (
    get_frame_index,
    get_ms_from_time_str,
    get_frame_index_from_ms,
    get_srt_timestamp,
    get_srt_timestamp_from_ms,
    extract_non_chinese_segments,
    is_blank_subtitle,
)
from videocr.exceptions import VideoValidationError


class TestTimeFunctions:
    """Test time-related utility functions."""
    
    def test_get_frame_index_minutes_seconds(self):
        """Test frame index calculation with MM:SS format."""
        assert get_frame_index("1:30", 30.0) == 2700
        assert get_frame_index("0:10", 30.0) == 300
        assert get_frame_index("2:00", 30.0) == 3600
    
    def test_get_frame_index_hours_minutes_seconds(self):
        """Test frame index calculation with HH:MM:SS format."""
        assert get_frame_index("1:00:00", 30.0) == 108000
        assert get_frame_index("0:01:30", 30.0) == 2700
        assert get_frame_index("0:00:10", 30.0) == 300
    
    def test_get_frame_index_invalid_format(self):
        """Test that invalid time format raises ValueError."""
        with pytest.raises(ValueError):
            get_frame_index("invalid", 30.0)
        with pytest.raises(ValueError):
            get_frame_index("1", 30.0)
    
    def test_get_ms_from_time_str_minutes_seconds(self):
        """Test millisecond conversion with MM:SS format."""
        assert get_ms_from_time_str("1:30") == 90000.0
        assert get_ms_from_time_str("0:10") == 10000.0
        assert get_ms_from_time_str("2:00") == 120000.0
    
    def test_get_ms_from_time_str_hours_minutes_seconds(self):
        """Test millisecond conversion with HH:MM:SS format."""
        assert get_ms_from_time_str("1:00:00") == 3600000.0
        assert get_ms_from_time_str("0:01:30") == 90000.0
        assert get_ms_from_time_str("0:00:10") == 10000.0
    
    def test_get_frame_index_from_ms(self):
        """Test getting frame index from timestamp."""
        timestamps = {0: 0.0, 30: 1000.0, 60: 2000.0}
        assert get_frame_index_from_ms(timestamps, 1000.0) == 30
        assert get_frame_index_from_ms(timestamps, 1500.0) == 30
        assert get_frame_index_from_ms(timestamps, 500.0) == 0
    
    def test_get_srt_timestamp(self):
        """Test SRT timestamp formatting."""
        assert get_srt_timestamp(0, 30.0, 0.0) == "00:00:00,000"
        assert get_srt_timestamp(30, 30.0, 0.0) == "00:00:01,000"
        assert get_srt_timestamp(90, 30.0, 0.0) == "00:00:03,000"
        assert get_srt_timestamp(0, 30.0, 1000.0) == "00:00:01,000"
    
    def test_get_srt_timestamp_from_ms(self):
        """Test SRT timestamp formatting from milliseconds."""
        assert get_srt_timestamp_from_ms(0) == "00:00:00,000"
        assert get_srt_timestamp_from_ms(1000) == "00:00:01,000"
        assert get_srt_timestamp_from_ms(61000) == "00:01:01,000"
        assert get_srt_timestamp_from_ms(3661000) == "01:01:01,000"


class TestTextProcessing:
    """Test text processing functions."""
    
    def test_extract_non_chinese_segments(self):
        """Test extraction of non-Chinese text segments."""
        text = "Hello世界Test"
        segments = extract_non_chinese_segments(text)
        
        assert len(segments) == 4
        assert segments[0] == ('non_chinese', 'Hello')
        assert segments[1] == ('chinese', '世')
        assert segments[2] == ('chinese', '界')
        assert segments[3] == ('non_chinese', 'Test')
    
    def test_extract_non_chinese_segments_all_english(self):
        """Test with all English text."""
        text = "Hello World"
        segments = extract_non_chinese_segments(text)
        
        assert len(segments) == 1
        assert segments[0] == ('non_chinese', 'Hello World')
    
    def test_extract_non_chinese_segments_all_chinese(self):
        """Test with all Chinese text."""
        text = "世界你好"
        segments = extract_non_chinese_segments(text)
        
        assert len(segments) == 4
        assert all(seg[0] == 'chinese' for seg in segments)
    
    def test_is_blank_subtitle_empty(self):
        """Test blank subtitle detection with empty string."""
        assert is_blank_subtitle("") is True
        assert is_blank_subtitle("   ") is True
        assert is_blank_subtitle(None) is True
    
    def test_is_blank_subtitle_too_short(self):
        """Test blank subtitle detection with very short text."""
        assert is_blank_subtitle("a") is True
        assert is_blank_subtitle("1") is True
    
    def test_is_blank_subtitle_common_patterns(self):
        """Test blank subtitle detection with common VLM patterns."""
        assert is_blank_subtitle("no text found") is True
        assert is_blank_subtitle("No text visible") is True
        assert is_blank_subtitle("This image shows a") is True
        assert is_blank_subtitle("I cannot see any text") is True
    
    def test_is_blank_subtitle_valid_text(self):
        """Test that valid text is not detected as blank."""
        assert is_blank_subtitle("Hello World") is False
        assert is_blank_subtitle("测试文本") is False
        assert is_blank_subtitle("This is a subtitle") is False
    
    def test_is_blank_subtitle_punctuation_only(self):
        """Test blank subtitle detection with punctuation only."""
        assert is_blank_subtitle("!!!") is True
        assert is_blank_subtitle("...") is True
        assert is_blank_subtitle("???") is True


class TestPathValidation:
    """Test path validation functions."""
    
    def test_safe_path_join_normal(self):
        """Test normal path joining."""
        from videocr_glm_cli import safe_path_join
        import os
        
        result = safe_path_join("/base", "subdir", "file.txt")
        expected = os.path.abspath(os.path.join("/base", "subdir", "file.txt"))
        assert result == expected
    
    def test_safe_path_join_traversal_attempt(self):
        """Test that path traversal is prevented."""
        from videocr_glm_cli import safe_path_join
        
        with pytest.raises(VideoValidationError):
            safe_path_join("/base", "../etc/passwd")
        
        with pytest.raises(VideoValidationError):
            safe_path_join("/base", "subdir/../../etc/passwd")
    
    def test_safe_path_join_absolute_path(self):
        """Test that absolute paths are handled correctly."""
        from videocr_glm_cli import safe_path_join
        
        with pytest.raises(VideoValidationError):
            safe_path_join("/base", "/etc/passwd")


class TestConfig:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test OCRConfig initialization."""
        from videocr.config import OCRConfig
        
        config = OCRConfig()
        assert config.log_response is False
        assert config.log_payload is False
        assert config.log_image is False
    
    def test_config_enable_logging(self):
        """Test enabling logging options."""
        from videocr.config import OCRConfig
        
        config = OCRConfig()
        config.enable_logging(response=True, payload=True)
        
        assert config.log_response is True
        assert config.log_payload is True
    
    def test_config_disable_logging(self):
        """Test disabling logging options."""
        from videocr.config import OCRConfig
        
        config = OCRConfig(log_response=True, log_payload=True)
        config.disable_logging()
        
        assert config.log_response is False
        assert config.log_payload is False
        assert config.log_image is False
    
    def test_config_repr(self):
        """Test config string representation."""
        from videocr.config import OCRConfig
        
        config = OCRConfig(log_response=True, log_payload=False)
        repr_str = repr(config)
        
        assert "OCRConfig" in repr_str
        assert "log_response=True" in repr_str
        assert "log_payload=False" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])