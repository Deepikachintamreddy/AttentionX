import numpy as np
import librosa
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import List


@dataclass
class PeakMoment:
    start_sec: float
    end_sec: float
    peak_sec: float
    attention_score: float
    rms_score: float
    zcr_score: float
    duration: float


def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _extract_wav(video_path: str) -> str:
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        _get_ffmpeg(), "-y", "-i", video_path,
        "-vn", "-ar", "22050", "-ac", "1",
        "-c:a", "pcm_s16le", tmp_wav,
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True, timeout=120)
    return tmp_wav


def analyze_audio(video_path: str, clip_duration: int = 60, top_n: int = 5) -> List[PeakMoment]:
    wav_path = _extract_wav(video_path)
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    hop_length = 512
    frame_length = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    window_frames = int(3 * sr / hop_length)
    rms_smooth = np.convolve(rms, np.ones(window_frames) / window_frames, mode="same")
    zcr_smooth = np.convolve(zcr, np.ones(window_frames) / window_frames, mode="same")

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    rms_norm = norm(rms_smooth)
    zcr_norm = norm(zcr_smooth)
    attention = 0.65 * rms_norm + 0.35 * zcr_norm

    total_duration = times[-1]

    # IMPORTANT: Zero out the first clip_duration seconds
    # so we never pick clips from the very beginning
    blackout_frames = int(clip_duration * sr / hop_length)
    attention_search = attention.copy()
    attention_search[:blackout_frames] = 0.0

    # Also zero out last clip_duration seconds
    if len(attention_search) > blackout_frames:
        attention_search[-blackout_frames:] = 0.0

    min_gap_frames = int(clip_duration * sr / hop_length)
    peaks = _find_peaks(attention_search, min_gap=min_gap_frames, top_n=top_n * 2)

    moments = []
    seen_windows = []

    for peak_frame in peaks:
        peak_sec = float(times[min(peak_frame, len(times) - 1)])

        half = clip_duration / 2
        start = max(0.0, peak_sec - half)
        end = min(total_duration, start + clip_duration)
        start = max(0.0, end - clip_duration)

        # Skip if window overlaps existing
        overlapping = False
        for s, e in seen_windows:
            overlap = min(end, e) - max(start, s)
            if overlap / clip_duration > 0.5:
                overlapping = True
                break
        if overlapping:
            continue

        # Use original attention score (not zeroed)
        score = float(attention[peak_frame]) * 100

        moments.append(PeakMoment(
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            peak_sec=round(peak_sec, 2),
            attention_score=round(score, 1),
            rms_score=round(float(rms_norm[peak_frame]) * 100, 1),
            zcr_score=round(float(zcr_norm[peak_frame]) * 100, 1),
            duration=round(end - start, 2),
        ))
        seen_windows.append((start, end))

        if len(moments) >= top_n:
            break

    # Fallback: if no moments found, use middle of video
    if not moments:
        mid = total_duration / 2
        start = max(0.0, mid - clip_duration / 2)
        end = min(total_duration, start + clip_duration)
        moments.append(PeakMoment(
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            peak_sec=round(mid, 2),
            attention_score=50.0,
            rms_score=50.0,
            zcr_score=50.0,
            duration=round(end - start, 2),
        ))

    moments.sort(key=lambda m: m.attention_score, reverse=True)
    return moments


def get_full_timeline(video_path: str, resolution_sec: float = 2.0):
    wav_path = _extract_wav(video_path)
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    window_frames = int(3 * sr / hop_length)

    def smooth_norm(x):
        s = np.convolve(x, np.ones(window_frames) / window_frames, mode="same")
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    attention = 0.65 * smooth_norm(rms) + 0.35 * smooth_norm(zcr)
    step = max(1, int(resolution_sec * sr / hop_length))
    return times[::step].tolist(), (attention[::step] * 100).tolist()


def _find_peaks(signal: np.ndarray, min_gap: int, top_n: int) -> List[int]:
    remaining = signal.copy()
    peaks = []
    for _ in range(top_n):
        if remaining.max() == 0:
            break
        idx = int(np.argmax(remaining))
        peaks.append(idx)
        lo = max(0, idx - min_gap)
        hi = min(len(remaining), idx + min_gap)
        remaining[lo:hi] = 0.0
    return peaks