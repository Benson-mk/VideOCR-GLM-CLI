from __future__ import annotations

import cv2
import fast_ssim
import wordninja_enhanced as wordninja
from pymediainfo import MediaInfo

from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .pyav_adapter import Capture, get_video_properties


class Video:
    """
    Main class for video processing and OCR operations.
    
    This class handles video file loading, frame extraction, OCR processing,
    and subtitle generation. It supports both constant frame rate (CFR) and
    variable frame rate (VFR) videos.
    
    Attributes:
        path: Path to the video file
        ollama_host: Ollama server host address
        ollama_port: Ollama server port
        ollama_model: Ollama model name for OCR
        ollama_timeout: Ollama API timeout in seconds
        frame_timestamps: Dictionary mapping frame indices to timestamps (VFR only)
        start_time_offset_ms: Time offset in milliseconds for VFR videos
        height: Video height in pixels
        width: Video width in pixels
        fps: Video frames per second
        num_frames: Total number of frames in video
        is_vfr: Whether video has variable frame rate
        lang: OCR language code
        use_fullframe: Whether to use full frame for OCR
        validated_zones: List of validated crop zones
        pred_frames_zone1: Predicted frames for zone 1
        pred_frames_zone2: Predicted frames for zone 2
        pred_subs: Generated subtitles
    """
    
    def __init__(self, path: str, ollama_host: str, ollama_port: int,
                 ollama_model: str, ollama_timeout: int, time_end: str | None = None):
        """
        Initialize Video object.
        
        Args:
            path: Path to the video file
            ollama_host: Ollama server host address
            ollama_port: Ollama server port
            ollama_model: Ollama model name for OCR
            ollama_timeout: Ollama API timeout in seconds
            time_end: Optional end time for processing (MM:SS or HH:MM:SS)
        """
        self.path = path
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_model = ollama_model
        self.ollama_timeout = ollama_timeout
        self.frame_timestamps = {}
        self.start_time_offset_ms = 0.0

        media_info = MediaInfo.parse(path)
        video_track = [t for t in media_info.tracks if t.track_type == 'Video'][0]

        initial_fps = float(video_track.frame_rate) if video_track.frame_rate else 0.0
        initial_num_frames = int(video_track.frame_count) if video_track.frame_count else 0

        self.is_vfr = (
            video_track.frame_rate_mode == 'VFR'
            or video_track.framerate_mode_original == 'VFR'
        )

        props = get_video_properties(self.path, self.is_vfr, time_end, initial_fps, initial_num_frames)
        self.height = props['height']
        self.width = props['width']
        self.fps = props['fps']
        self.num_frames = props['num_frames']
        self.start_time_offset_ms = props['start_time_offset_ms']
        self.frame_timestamps = props['frame_timestamps']

        self.lang = ''
        self.use_fullframe = False
        self.validated_zones = []
        self.pred_frames_zone1 = []
        self.pred_frames_zone2 = []
        self.pred_subs = []

    def run_ocr(self, lang: str, time_start: str, time_end: str, use_fullframe: bool,
                brightness_threshold: int, ssim_threshold: int, subtitle_position: str,
                frames_to_skip: int, crop_zones: list[dict], ocr_image_max_width: int,
                normalize_to_simplified_chinese: bool = True) -> None:
        """
        Run OCR processing on video frames.
        
        This method extracts frames from the video, applies preprocessing
        (cropping, brightness threshold, resizing), performs OCR using GLM-OCR,
        and stores the results for subtitle generation.
        
        Args:
            lang: OCR language code
            time_start: Start time for processing (MM:SS or HH:MM:SS)
            time_end: End time for processing (MM:SS or HH:MM:SS)
            use_fullframe: Use full frame for OCR instead of crop zones
            brightness_threshold: Brightness threshold for image preprocessing
            ssim_threshold: SSIM similarity threshold for frame skipping (0-100)
            subtitle_position: Subtitle position alignment (center, left, right, any)
            frames_to_skip: Number of frames to skip between OCR operations
            crop_zones: List of crop zone dictionaries
            ocr_image_max_width: Maximum image width for OCR
            normalize_to_simplified_chinese: Convert Traditional Chinese to Simplified
        
        Raises:
            ValueError: If time_start is after time_end or crop zones are invalid
        """
        
        ssim_threshold = float(ssim_threshold / 100)
        self.lang = lang
        self.use_fullframe = use_fullframe
        self.validated_zones = []
        self.pred_frames_zone1 = []
        self.pred_frames_zone2 = []

        # Calculate frame range
        if self.is_vfr:
            if time_start:
                start_target_ms = utils.get_ms_from_time_str(time_start) + self.start_time_offset_ms
                ocr_start = utils.get_frame_index_from_ms(self.frame_timestamps, start_target_ms)
            else:
                ocr_start = 0
            if time_end:
                end_target_ms = utils.get_ms_from_time_str(time_end) + self.start_time_offset_ms
                ocr_end = utils.get_frame_index_from_ms(self.frame_timestamps, end_target_ms)
            else:
                ocr_end = self.num_frames
        else:
            ocr_start = utils.get_frame_index(time_start, self.fps) if time_start else 0
            ocr_end = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames

        if ocr_end < ocr_start:
            raise ValueError('time_start is later than time_end')
        num_ocr_frames = ocr_end - ocr_start

        # Validate crop zones
        for zone in crop_zones:
            if zone['y'] >= self.height:
                raise ValueError(f"Crop Y position ({zone['y']}) is outside video height ({self.height}).")
            if zone['x'] >= self.width:
                raise ValueError(f"Crop X position ({zone['x']}) is outside video width ({self.width}).")
            self.validated_zones.append({
                'x_start': zone['x'], 'y_start': zone['y'],
                'x_end': zone['x'] + zone['width'], 'y_end': zone['y'] + zone['height'],
                'midpoint_y': zone['y'] + (zone['height'] / 2)
            })

        # Process frames
        with Capture(self.path) as v:
            if ocr_start > 0:
                for i in range(ocr_start):
                    v.grab()
                    print(f"\rAdvancing to frame {i + 1}/{ocr_start}", end="", flush=True)
                print()

            prev_samples = [None] * len(self.validated_zones) if self.validated_zones else [None]
            modulo = frames_to_skip + 1
            frame_predictions_by_zone = {0: {}, 1: {}}
            
            # Track blank frames for each zone to properly set subtitle end times
            blank_frames_by_zone = {0: [], 1: []}

            for i in range(num_ocr_frames):
                print(f"\rProcessing frame {i + 1} of {num_ocr_frames}", end="", flush=True)
                
                if i % modulo == 0:
                    read_success, frame = v.read()
                    if not read_success:
                        continue

                    # Determine regions to process
                    images_to_process = []
                    if use_fullframe:
                        images_to_process.append({'image': frame, 'zone_idx': 0})
                    elif self.validated_zones:
                        for idx, zone in enumerate(self.validated_zones):
                            images_to_process.append({
                                'image': frame[zone['y_start']:zone['y_end'], zone['x_start']:zone['x_end']],
                                'zone_idx': idx
                            })
                    else:
                        # Default to bottom third for subtitles
                        images_to_process.append({'image': frame[2 * self.height // 3:, :], 'zone_idx': 0})

                    for item in images_to_process:
                        img = item['image']
                        zone_idx = item['zone_idx']

                        # Resize if needed
                        if ocr_image_max_width and img.shape[1] > ocr_image_max_width:
                            original_height, original_width = img.shape[:2]
                            scale_ratio = ocr_image_max_width / original_width
                            new_height = int(original_height * scale_ratio)
                            img = cv2.resize(img, (ocr_image_max_width, new_height), interpolation=cv2.INTER_AREA)

                        # Apply brightness threshold
                        if brightness_threshold:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
                            img = cv2.bitwise_and(img, img, mask=mask)

                        # SSIM check to skip similar frames
                        if ssim_threshold < 1:
                            w = img.shape[1]
                            if subtitle_position == "center":
                                w_margin = int(w * 0.35)
                                sample = img[:, w_margin:w - w_margin]
                            elif subtitle_position == "left":
                                sample = img[:, :int(w * 0.3)]
                            elif subtitle_position == "right":
                                sample = img[:, int(w * 0.7):]
                            elif subtitle_position == "any":
                                sample = img
                            else:
                                sample = img

                            if prev_samples[zone_idx] is not None:
                                score = fast_ssim.ssim(prev_samples[zone_idx], sample, data_range=255)
                                if score > ssim_threshold:
                                    prev_samples[zone_idx] = sample
                                    continue
                            prev_samples[zone_idx] = sample

                        # Call GLM-OCR
                        frame_index = i + ocr_start
                        try:
                            ocr_text, is_valid = utils.call_glm_ocr(
                                img, self.ollama_host, self.ollama_port,
                                self.ollama_model, self.ollama_timeout, self.lang,
                                frame_index=frame_index, zone_idx=zone_idx,
                                log_payload=True, log_response=True
                            )
                            
                            if is_valid and ocr_text:
                                predicted_frame = PredictedFrames(
                                    frame_index, ocr_text, zone_idx, self.lang,
                                    normalize_to_simplified_chinese
                                )
                                frame_predictions_by_zone[zone_idx][frame_index] = predicted_frame
                                print(f"\rProcessing frame {i + 1} of {num_ocr_frames} (text: {len(ocr_text)} chars)          ", end="", flush=True)
                            else:
                                # Blank frame - no subtitle detected, track it for proper end time calculation
                                blank_frames_by_zone[zone_idx].append(frame_index)
                                print(f"\rProcessing frame {i + 1} of {num_ocr_frames} (blank)          ", end="", flush=True)
                                
                        except Exception as e:
                            print(f"\nWarning: OCR failed for frame {frame_index}: {e}", flush=True)

                            frame_index = i + ocr_start
                else:
                    v.grab()

        print()

        # Set end indices using blank frames to determine actual subtitle boundaries
        for zone_idx in frame_predictions_by_zone:
            frames = sorted(frame_predictions_by_zone[zone_idx].values(), key=lambda f: f.start_index)
            blank_frames = sorted(blank_frames_by_zone.get(zone_idx, []))
            
            if not frames:
                continue
            
            for i in range(len(frames) - 1):
                current_frame = frames[i]
                next_frame = frames[i + 1]
                
                # Find blank frames between current subtitle and next subtitle
                intervening_blanks = [
                    bf for bf in blank_frames 
                    if current_frame.start_index < bf < next_frame.start_index
                ]
                
                if intervening_blanks:
                    # Blank frames detected - subtitle should end before the first blank frame
                    current_frame.end_index = intervening_blanks[0] - 1
                else:
                    # No blank frames detected between subtitles, use original logic
                    current_frame.end_index = next_frame.start_index - 1
            
            if frames:
                frames[-1].end_index = ocr_end - 1
            
            frame_predictions_by_zone[zone_idx] = frames

        self.pred_frames_zone1 = frame_predictions_by_zone.get(0, [])
        self.pred_frames_zone2 = frame_predictions_by_zone.get(1, [])

    def get_subtitles(self, sim_threshold: int, max_merge_gap_sec: float, lang: str,
                      post_processing: bool, min_subtitle_duration_sec: float) -> str:
        """
        Generate and return subtitles in SRT format.
        
        This method processes the OCR results, groups similar frames into subtitles,
        applies post-processing if enabled, filters by minimum duration, and formats
        the output as SRT subtitle text.
        
        Args:
            sim_threshold: Similarity threshold for merging subtitles (0-100)
            max_merge_gap_sec: Maximum time gap in seconds to merge similar subtitles
            lang: OCR language code
            post_processing: Enable text post-processing (e.g., word segmentation)
            min_subtitle_duration_sec: Minimum subtitle duration in seconds
        
        Returns:
            SRT-formatted subtitle string
        """
        self._generate_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec)

        srt_lines = []
        for i, sub in enumerate(self.pred_subs, 1):
            start_time, end_time = self._get_srt_timestamps(sub)
            srt_lines.append(f'{i}\n{start_time} --> {end_time}\n{sub.text}\n\n')

        return ''.join(srt_lines)

    def _generate_subtitles(self, sim_threshold: int, max_merge_gap_sec: float, lang: str,
                            post_processing: bool, min_subtitle_duration_sec: float) -> None:
        print("Generating subtitles...", flush=True)

        subs_zone1 = self._process_single_zone(
            self.pred_frames_zone1, sim_threshold, max_merge_gap_sec, lang,
            post_processing, min_subtitle_duration_sec
        )
        subs_zone2 = self._process_single_zone(
            self.pred_frames_zone2, sim_threshold, max_merge_gap_sec, lang,
            post_processing, min_subtitle_duration_sec
        )

        if subs_zone1 and not subs_zone2:
            self.pred_subs = subs_zone1
        elif not subs_zone1 and subs_zone2:
            self.pred_subs = subs_zone2
        elif subs_zone1 and subs_zone2:
            self.pred_subs = self._merge_dual_zone_subtitles(subs_zone1, subs_zone2)
        else:
            self.pred_subs = []

    def _process_single_zone(self, pred_frames: list[PredictedFrames], sim_threshold: int,
                             max_merge_gap_sec: float, lang: str, post_processing: bool,
                             min_subtitle_duration_sec: float) -> list[PredictedSubtitle]:
        if not pred_frames:
            return []

        language_mapping = {"en": "en", "fr": "fr", "german": "de", "it": "it", "es": "es", "pt": "pt"}
        language_model = None
        if post_processing and lang in language_mapping:
            language_model = wordninja.LanguageModel(language=language_mapping[lang])

        subs = []
        for frame in sorted(pred_frames, key=lambda f: f.start_index):
            new_sub = PredictedSubtitle([frame], frame.zone_index, sim_threshold, lang, language_model)
            if not new_sub.text:
                continue

            if subs:
                last_sub = subs[-1]
                if self._is_gap_mergeable(last_sub, new_sub, max_merge_gap_sec) and last_sub.is_similar_to(new_sub):
                    last_sub.frames.extend(new_sub.frames)
                    last_sub.frames.sort(key=lambda f: f.start_index)
                else:
                    subs.append(new_sub)
            else:
                subs.append(new_sub)

        for sub in subs:
            sub.finalize_text(post_processing)

        filtered_subs = [
            sub for sub in subs if self._get_subtitle_duration_sec(sub) >= min_subtitle_duration_sec
        ]

        if not filtered_subs:
            return []

        cleaned_subs = [filtered_subs[0]]
        for next_sub in filtered_subs[1:]:
            last_sub = cleaned_subs[-1]
            if self._is_gap_mergeable(last_sub, next_sub, max_merge_gap_sec) and last_sub.is_similar_to(next_sub):
                last_sub.frames.extend(next_sub.frames)
                last_sub.frames.sort(key=lambda f: f.start_index)
                last_sub.finalize_text(post_processing)
            else:
                cleaned_subs.append(next_sub)

        return cleaned_subs

    def _merge_dual_zone_subtitles(self, subs1: list[PredictedSubtitle], subs2: list[PredictedSubtitle]) -> list[PredictedSubtitle]:
        all_subs = sorted(subs1 + subs2, key=lambda s: s.index_start)
        if not all_subs:
            return []

        merged_subs = [all_subs[0]]
        for current_sub in all_subs[1:]:
            last_sub = merged_subs[-1]
            if current_sub.index_start <= last_sub.index_end:
                last_zone_info = self.validated_zones[last_sub.zone_index]
                current_zone_info = self.validated_zones[current_sub.zone_index]
                if current_zone_info['midpoint_y'] < last_zone_info['midpoint_y']:
                    last_sub.text = f"{current_sub.text}\n{last_sub.text}"
                else:
                    last_sub.text = f"{last_sub.text}\n{current_sub.text}"
                last_sub.frames.extend(current_sub.frames)
                last_sub.frames.sort(key=lambda f: f.start_index)
            else:
                merged_subs.append(current_sub)
        return merged_subs

    def _get_srt_timestamps(self, sub: PredictedSubtitle) -> tuple[str, str]:
        if self.is_vfr:
            start_ms, end_ms = self._get_subtitle_ms_times(sub)
            return utils.get_srt_timestamp_from_ms(start_ms), utils.get_srt_timestamp_from_ms(end_ms)
        else:
            start_time = utils.get_srt_timestamp(sub.index_start, self.fps, self.start_time_offset_ms)
            end_time = utils.get_srt_timestamp(sub.index_end + 1, self.fps, self.start_time_offset_ms)
            return start_time, end_time

    def _get_subtitle_ms_times(self, sub: PredictedSubtitle) -> tuple[float, float]:
        first_frame_ms = self.frame_timestamps.get(0, 0)
        correction_delta = first_frame_ms - self.start_time_offset_ms
        start_time_ms = self.frame_timestamps.get(sub.index_start, 0)
        end_time_ms = self.frame_timestamps.get(sub.index_end + 1)
        if end_time_ms is None:
            end_time_ms = self.frame_timestamps.get(sub.index_end, 0) + (1000 / self.fps)
        return start_time_ms - correction_delta, end_time_ms - correction_delta

    def _get_subtitle_duration_sec(self, sub: PredictedSubtitle) -> float:
        if self.is_vfr:
            start_ms, end_ms = self._get_subtitle_ms_times(sub)
            return (end_ms - start_ms) / 1000
        else:
            return (sub.index_end + 1 - sub.index_start) / self.fps

    def _is_gap_mergeable(self, last_sub: PredictedSubtitle, next_sub: PredictedSubtitle, max_merge_gap_sec: float) -> bool:
        if self.is_vfr:
            _, last_end_ms = self._get_subtitle_ms_times(last_sub)
            next_start_ms, _ = self._get_subtitle_ms_times(next_sub)
            gap_ms = next_start_ms - last_end_ms
            return gap_ms <= (max_merge_gap_sec * 1000)
        else:
            max_frame_merge_diff = int(max_merge_gap_sec * self.fps) + 1
            gap_frames = next_sub.index_start - last_sub.index_end
            return gap_frames <= max_frame_merge_diff