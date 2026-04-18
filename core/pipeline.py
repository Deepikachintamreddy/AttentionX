import os
import time
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Optional

from core.audio_analyzer import analyze_audio, get_full_timeline, PeakMoment
from core.gemini_analyzer import analyze_clip, GeminiAnalysis


@dataclass
class ClipResult:
    rank: int
    peak: PeakMoment
    analysis: GeminiAnalysis
    output_path: str
    processing_time_sec: float


@dataclass
class PipelineResult:
    clips: List[ClipResult]
    timeline_times: List[float]
    timeline_scores: List[float]
    total_time_sec: float


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _extract_clip(video_path, start, end, out_path):
    ffmpeg = _get_ffmpeg()
    subprocess.run([
        ffmpeg, "-y", "-ss", str(start), "-to", str(end),
        "-i", video_path, "-c", "copy", out_path,
        "-loglevel", "error"
    ], check=True, timeout=60)


def _detect_face_cx(frame, use_mediapipe, detector, src_w):
    import cv2
    h, w = frame.shape[:2]
    if use_mediapipe and detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            return int((bbox.xmin + bbox.width / 2) * w)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            x, _, fw, _ = faces[0]
            return x + fw // 2
    return None


def _crop_to_vertical(input_path, output_path):
    import cv2
    try:
        import mediapipe as mp
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        use_mediapipe = True
    except Exception:
        use_mediapipe = False
        face_detection = None

    cap = cv2.VideoCapture(input_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w, out_h = 1080, 1920
    crop_w = int(src_h * 9 / 16)
    if crop_w > src_w:
        crop_w = src_w

    # Sample frames to find face position
    sample_positions = []
    sample_step = max(1, total_frames // 20)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step == 0:
            cx = _detect_face_cx(frame, use_mediapipe, face_detection, src_w)
            if cx is not None:
                sample_positions.append(cx)
        frame_idx += 1
    cap.release()

    face_cx = int(np.median(sample_positions)) if sample_positions else src_w // 2

    half = crop_w // 2
    x_start = max(0, min(src_w - crop_w, face_cx - half))

    ffmpeg = _get_ffmpeg()
    result = subprocess.run([
        ffmpeg, "-y", "-i", input_path,
        "-vf", f"crop={crop_w}:{src_h}:{x_start}:0,scale={out_w}:{out_h}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", output_path, "-loglevel", "error"
    ], timeout=120, capture_output=True, text=True)

    if result.returncode != 0:
        x_center = (src_w - crop_w) // 2
        subprocess.run([
            ffmpeg, "-y", "-i", input_path,
            "-vf", f"crop={crop_w}:{src_h}:{x_center}:0,scale={out_w}:{out_h}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", output_path, "-loglevel", "error"
        ], check=True, timeout=120)


def _burn_hook(input_path, output_path, hook_text, clip_title):
    ffmpeg = _get_ffmpeg()
    safe_title = clip_title.replace("'", "").replace(":", "").replace(",", "")[:40]
    safe_hook  = hook_text.replace("'", "").replace(":", "").replace(",", "")[:60]

    filter_str = (
        f"drawbox=x=0:y=h*0.28:w=iw:h=h*0.22:"
        f"color=black@0.7:t=fill:enable='between(t,0,3)',"
        f"drawtext=text='{safe_title}':fontsize=54:fontcolor=yellow:"
        f"x=(w-text_w)/2:y=h*0.31:enable='between(t,0,3)',"
        f"drawtext=text='{safe_hook}':fontsize=36:fontcolor=white:"
        f"x=(w-text_w)/2:y=h*0.38:enable='between(t,0,3)'"
    )

    result = subprocess.run([
        ffmpeg, "-y", "-i", input_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", output_path, "-loglevel", "error"
    ], timeout=120, capture_output=True, text=True)

    if result.returncode != 0:
        subprocess.run([
            ffmpeg, "-y", "-i", input_path,
            "-c", "copy", output_path, "-loglevel", "error"
        ], check=True)


def run_pipeline(
    video_path: str,
    output_dir: str,
    gemini_api_key: str,
    clip_duration: int = 30,
    top_n_peaks: int = 3,
    top_n_clips: int = 1,
    whisper_model: str = "tiny",
    hook_type: str = "curiosity",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> PipelineResult:

    os.makedirs(output_dir, exist_ok=True)
    total_start = time.time()

    def progress(stage, pct):
        if on_progress:
            on_progress(stage, pct)

    # Stage 1: Audio analysis
    progress("Analysing audio for emotional peaks…", 0.05)
    peaks = analyze_audio(video_path, clip_duration=clip_duration, top_n=top_n_peaks)
    timeline_times, timeline_scores = get_full_timeline(video_path)
    progress("Audio analysis complete", 0.20)

    selected_peaks = peaks[:top_n_clips]
    clip_results = []

    for i, peak in enumerate(selected_peaks):

        # Extract clip
        progress(f"Extracting clip {i+1}/{len(selected_peaks)}…", 0.25 + 0.10*i)
        raw_clip = os.path.join(output_dir, f"raw_{i}.mp4")
        _extract_clip(video_path, peak.start_sec, peak.end_sec, raw_clip)

        # Gemini scoring
        progress(f"Scoring with Gemini AI…", 0.40 + 0.10*i)
        dummy_segments = [{
            "start": peak.start_sec,
            "end": peak.end_sec,
            "text": f"High energy moment at {peak.peak_sec:.0f} seconds. Audio attention score {peak.attention_score:.0f} out of 100."
        }]
        analysis = analyze_clip(
            transcript_segments=dummy_segments,
            window_start=peak.start_sec,
            window_end=peak.end_sec,
            api_key=gemini_api_key,
        )

        # Face-tracked vertical crop
        progress(f"Smart cropping to 9:16 with face tracking…", 0.55 + 0.10*i)
        vertical_clip = os.path.join(output_dir, f"vertical_{i}.mp4")
        try:
            _crop_to_vertical(raw_clip, vertical_clip)
        except Exception as e:
            print(f"Crop error: {e}, using raw")
            vertical_clip = raw_clip

        # Burn hook overlay
        progress(f"Burning hook overlay…", 0.80 + 0.10*i)
        final_clip = os.path.join(output_dir, f"clip_{i+1:02d}.mp4")
        hook = getattr(analysis.hooks, hook_type, analysis.hooks.curiosity)
        try:
            _burn_hook(vertical_clip, final_clip, hook, analysis.clip_title)
        except Exception as e:
            print(f"Hook burn error: {e}")
            final_clip = vertical_clip

        # Cleanup
        for f in [raw_clip, vertical_clip]:
            if os.path.exists(f) and f != final_clip:
                try:
                    os.remove(f)
                except Exception:
                    pass

        clip_results.append(ClipResult(
            rank=i + 1,
            peak=peak,
            analysis=analysis,
            output_path=final_clip,
            processing_time_sec=round(time.time() - total_start, 1),
        ))

    progress("All clips ready!", 1.0)

    return PipelineResult(
        clips=clip_results,
        timeline_times=timeline_times,
        timeline_scores=timeline_scores,
        total_time_sec=round(time.time() - total_start, 1),
    )