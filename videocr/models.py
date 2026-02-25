from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import wordninja_enhanced as wordninja
from opencc import OpenCC
from thefuzz import fuzz

from . import utils


@dataclass
class PredictedText:
    """GLM-OCR doesn't return bounding boxes, so we simplify this."""
    __slots__ = ('confidence', 'text')
    confidence: float
    text: str


class PredictedFrames:
    """Stores OCR result for a single frame."""
    __slots__ = ('start_index', 'end_index', 'zone_index', 'confidence', 'text')
    
    # Class variable (NOT in __slots__)
    _converter = OpenCC('t2s')

    start_index: int
    end_index: int
    zone_index: int
    confidence: float
    text: str

    def __init__(self, index: int, text: str, zone_index: int, lang: str,
                 normalize_to_simplified_chinese: bool = True):
        self.start_index = index
        self.end_index = index
        self.zone_index = zone_index
        self.text = text.strip() if text else ''
        
        # GLM-OCR doesn't return confidence, so we use a default
        # Empty text = 0 confidence (will be filtered out)
        self.confidence = 100.0 if self.text else 0.0

        if normalize_to_simplified_chinese and lang == "ch" and self.text:
            self.text = self._converter.convert(self.text)


class PredictedSubtitle:
    """Groups similar frames into a subtitle."""
    __slots__ = ('frames', 'zone_index', 'sim_threshold', 'text', 'lang', '_language_model')
    
    frames: list[PredictedFrames]
    zone_index: int
    sim_threshold: int
    text: str
    lang: str
    _language_model: wordninja.LanguageModel | None

    def __init__(self, frames: list[PredictedFrames], zone_index: int, sim_threshold: int,
                 lang: str, language_model: wordninja.LanguageModel | None = None):
        self.frames = [f for f in frames if f.confidence > 0]
        self.frames.sort(key=lambda frame: frame.start_index)
        self.zone_index = zone_index
        self.sim_threshold = sim_threshold
        self.lang = lang
        self._language_model = language_model

        if self.frames:
            self.text = max(self.frames, key=lambda f: f.confidence).text
        else:
            self.text = ''

    @property
    def index_start(self) -> int:
        return self.frames[0].start_index if self.frames else 0

    @property
    def index_end(self) -> int:
        return self.frames[-1].end_index if self.frames else 0

    def is_similar_to(self, other: PredictedSubtitle) -> bool:
        return fuzz.ratio(self.text.replace(' ', ''), other.text.replace(' ', '')) >= self.sim_threshold

    def __repr__(self):
        return f'{self.index_start} - {self.index_end}. {self.text}'

    def finalize_text(self, post_processing: bool) -> None:
        text_counts = Counter()
        text_confidences = defaultdict(list)

        for frame in self.frames:
            text_counts[frame.text] += 1
            text_confidences[frame.text].append(frame.confidence)

        max_count = max(text_counts.values())
        candidates = [text for text, count in text_counts.items() if count == max_count]

        if len(candidates) == 1:
            final_text = candidates[0]
        else:
            final_text = max(
                candidates,
                key=lambda t: sum(text_confidences[t]) / len(text_confidences[t])
            )

        if post_processing:
            if self.lang in ("en", "fr", "german", "it", "es", "pt"):
                final_text = self._language_model.rejoin(final_text)
            elif self.lang == "ch":
                segments = utils.extract_non_chinese_segments(final_text)
                rebuilt_text = ''
                for typ, seg in segments:
                    if typ == 'non_chinese':
                        rebuilt_text += wordninja.rejoin(seg)
                    else:
                        rebuilt_text += seg
                final_text = rebuilt_text

        self.text = final_text