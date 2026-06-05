"""
Video extraction utilities.
Pulls frames at 1 FPS and raw audio at 16kHz — matching Gemma 4 12B's native input format.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from PIL import Image
import base64
import io


def extract_frames(video_path: str, fps: int = 1, max_frames: int = 60) -> list[dict]:
    """
    Extract frames from video at given FPS.
    Returns list of {timestamp_s, base64_jpeg} dicts.
    """
    out_dir = tempfile.mkdtemp(prefix="tsi_frames_")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale=640:-1",
        "-q:v", "3",
        f"{out_dir}/frame_%04d.jpg",
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    frames = []
    frame_files = sorted(Path(out_dir).glob("frame_*.jpg"))[:max_frames]

    for i, f in enumerate(frame_files):
        timestamp = i / fps
        img = Image.open(f)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        frames.append({"timestamp_s": timestamp, "b64": b64, "path": str(f)})

    return frames


def extract_audio_segment(video_path: str, start_s: float, duration_s: float = 30.0) -> str:
    """
    Extract a raw audio segment as a temp WAV file (16kHz mono).
    Gemma 4 12B ingests 16kHz audio projected directly into token space.
    """
    out_path = tempfile.mktemp(suffix=".wav", prefix="tsi_audio_")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(start_s),
        "-t", str(duration_s),
        "-ar", "16000",   # 16kHz — Gemma 4 12B's native rate
        "-ac", "1",       # mono
        "-f", "wav",
        out_path,
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    return out_path


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def audio_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
