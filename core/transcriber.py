import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List


@dataclass
class WordSegment:
    word: str
    start: float
    end: float


@dataclass
class TranscriptResult:
    full_text: str
    words: List[WordSegment]
    segments: List[dict]


def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def transcribe_segment(video_path: str, start_sec: float, end_sec: float,
                       model_size: str = "tiny") -> TranscriptResult:
    return _fallback_transcript(start_sec, end_sec)


def _fallback_transcript(start_sec: float, end_sec: float) -> TranscriptResult:
    duration = end_sec - start_sec
    dummy_text = "This is a key insight from the video that viewers will find valuable and engaging."
    words = []
    word_list = dummy_text.split()
    step = duration / len(word_list)
    for i, w in enumerate(word_list):
        words.append(WordSegment(
            word=w,
            start=start_sec + i * step,
            end=start_sec + (i + 1) * step,
        ))
    return TranscriptResult(
        full_text=dummy_text,
        words=words,
        segments=[{"start": start_sec, "end": end_sec, "text": dummy_text}],
    )