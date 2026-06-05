"""
Core inspection logic.
Sends frames + audio to Gemma 4 12B via Ollama in a single multimodal prompt.

Key insight: because Gemma 4 12B has no separate encoders, audio and video tokens
share the same RoPE positional space. The model can reason about temporal alignment
between what it sees and what it hears without any intermediate representation.
"""

import json
import ollama
from extract import extract_frames, extract_audio_segment, get_video_duration, audio_to_base64


MODEL = "gemma4:12b"

SYSTEM_PROMPT = """You are a temporal sync analyzer. You receive video frames (with timestamps)
and the corresponding audio from the same time window.

Your job: determine whether the audio and video are temporally synchronized.

Focus on:
1. Lip sync — do mouth movements match the phonemes/words in the audio?
2. Action sync — do physical actions (clapping, typing, door closing) match audio events?
3. Emotional sync — does facial expression match vocal tone?

Respond ONLY with valid JSON in this exact schema:
{
  "in_sync": boolean,
  "confidence": "high" | "medium" | "low",
  "desync_detected_at_s": number | null,
  "desync_type": "lip_sync" | "action_sync" | "emotional_sync" | null,
  "estimated_offset_ms": number | null,
  "reasoning": "one concise sentence",
  "suspicious_frames": [timestamp_s, ...]
}"""


def build_prompt(frames: list[dict], window_start_s: float, window_end_s: float) -> str:
    frame_list = "\n".join(
        f"  - Frame at t={f['timestamp_s']:.1f}s" for f in frames
    )
    return (
        f"Analyze this {window_end_s - window_start_s:.0f}s video segment "
        f"(t={window_start_s:.1f}s to t={window_end_s:.1f}s).\n\n"
        f"Frames provided:\n{frame_list}\n\n"
        "The audio for this exact window is also provided. "
        "Determine if audio and video are in sync."
    )


def inspect_window(
    video_path: str,
    window_start_s: float = 0.0,
    window_duration_s: float = 30.0,
    fps: int = 1,
) -> dict:
    """
    Inspect a single time window of a video for A/V sync.
    Returns the parsed JSON result from Gemma 4 12B.
    """
    window_end_s = window_start_s + window_duration_s

    # Extract frames for this window
    frames = extract_frames(video_path, fps=fps, max_frames=int(window_duration_s * fps))
    # Filter to window
    frames = [f for f in frames if window_start_s <= f["timestamp_s"] < window_end_s]

    # Extract audio for this window (16kHz WAV — Gemma's native format)
    audio_path = extract_audio_segment(video_path, window_start_s, window_duration_s)
    audio_b64 = audio_to_base64(audio_path)

    # Build multimodal message: images + audio + text in one prompt
    # Gemma 4 12B processes all modalities through the same backbone
    images = [f["b64"] for f in frames]

    prompt_text = build_prompt(frames, window_start_s, window_end_s)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt_text,
                "images": images,
            },
        ],
        options={"temperature": 0.1},  # low temp for deterministic analysis
    )

    raw = response["message"]["content"].strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "in_sync": None,
            "confidence": "low",
            "desync_detected_at_s": None,
            "desync_type": None,
            "estimated_offset_ms": None,
            "reasoning": raw[:300],
            "suspicious_frames": [],
            "parse_error": True,
        }

    result["window_start_s"] = window_start_s
    result["window_end_s"] = window_end_s
    return result


def inspect_full_video(
    video_path: str,
    window_duration_s: float = 30.0,
    fps: int = 1,
    progress_cb=None,
) -> list[dict]:
    """
    Slide a window across the full video and inspect each segment.
    Returns list of per-window results.
    """
    duration = get_video_duration(video_path)
    results = []
    start = 0.0
    window_idx = 0
    total_windows = int(duration / window_duration_s) + 1

    while start < duration:
        actual_duration = min(window_duration_s, duration - start)
        if actual_duration < 5:
            break

        if progress_cb:
            progress_cb(window_idx / total_windows, f"Analyzing t={start:.0f}s–{start+actual_duration:.0f}s…")

        result = inspect_window(video_path, start, actual_duration, fps)
        results.append(result)
        start += window_duration_s
        window_idx += 1

    return results


def summarize_results(results: list[dict]) -> dict:
    """Aggregate per-window results into a video-level summary."""
    desync_windows = [r for r in results if not r.get("in_sync")]
    total = len(results)

    return {
        "total_windows": total,
        "desync_windows": len(desync_windows),
        "sync_score": round((total - len(desync_windows)) / max(total, 1) * 100, 1),
        "verdict": "IN SYNC" if len(desync_windows) == 0 else (
            "MINOR DESYNC" if len(desync_windows) / max(total, 1) < 0.3 else "SIGNIFICANT DESYNC"
        ),
        "desync_timestamps": [
            r["desync_detected_at_s"] for r in desync_windows
            if r.get("desync_detected_at_s") is not None
        ],
        "windows": results,
    }
