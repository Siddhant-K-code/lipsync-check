"""
Temporal Sync Inspector — Gradio UI
Powered by Gemma 4 12B (Ollama, local, no cloud)
"""

import gradio as gr
import json
import tempfile
import os
from inspector import inspect_full_video, inspect_window, summarize_results
from extract import get_video_duration


def format_summary(summary: dict) -> str:
    verdict_color = {
        "IN SYNC": "🟢",
        "MINOR DESYNC": "🟡",
        "SIGNIFICANT DESYNC": "🔴",
    }.get(summary["verdict"], "⚪")

    lines = [
        f"## {verdict_color} {summary['verdict']}",
        f"**Sync score:** {summary['sync_score']}%",
        f"**Windows analyzed:** {summary['total_windows']}",
        f"**Desync windows:** {summary['desync_windows']}",
    ]

    if summary["desync_timestamps"]:
        ts = ", ".join(f"{t:.1f}s" for t in summary["desync_timestamps"])
        lines.append(f"**Desync detected at:** {ts}")

    lines.append("\n---\n### Per-window breakdown")
    for w in summary["windows"]:
        icon = "✅" if w.get("in_sync") else "❌"
        conf = w.get("confidence", "?")
        reason = w.get("reasoning", "")
        dtype = w.get("desync_type", "")
        offset = w.get("estimated_offset_ms")
        offset_str = f" (~{offset}ms offset)" if offset else ""
        dtype_str = f" [{dtype}]" if dtype else ""
        lines.append(
            f"{icon} **t={w['window_start_s']:.0f}s–{w['window_end_s']:.0f}s** "
            f"[{conf} confidence]{dtype_str}{offset_str}  \n"
            f"  _{reason}_"
        )

    return "\n".join(lines)


def run_quick_check(video_file, window_start, window_duration, progress=gr.Progress()):
    if video_file is None:
        return "Upload a video first.", "{}"

    progress(0, desc="Extracting frames and audio…")
    try:
        result = inspect_window(
            video_file,
            window_start_s=float(window_start),
            window_duration_s=float(window_duration),
            fps=1,
        )
        summary = {
            "total_windows": 1,
            "desync_windows": 0 if result.get("in_sync") else 1,
            "sync_score": 100.0 if result.get("in_sync") else 0.0,
            "verdict": "IN SYNC" if result.get("in_sync") else "SIGNIFICANT DESYNC",
            "desync_timestamps": [result.get("desync_detected_at_s")] if not result.get("in_sync") else [],
            "windows": [result],
        }
        progress(1.0, desc="Done")
        return format_summary(summary), json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}", "{}"


def run_full_analysis(video_file, window_duration, fps, progress=gr.Progress()):
    if video_file is None:
        return "Upload a video first.", "{}"

    try:
        duration = get_video_duration(video_file)
    except Exception as e:
        return f"Could not read video: {e}", "{}"

    def progress_cb(pct, msg):
        progress(pct, desc=msg)

    try:
        results = inspect_full_video(
            video_file,
            window_duration_s=float(window_duration),
            fps=int(fps),
            progress_cb=progress_cb,
        )
        summary = summarize_results(results)
        progress(1.0, desc="Done")
        return format_summary(summary), json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error: {e}", "{}"


with gr.Blocks(
    title="Temporal Sync Inspector",
    theme=gr.themes.Default(primary_hue="violet"),
    css=".output-markdown { font-size: 0.95rem; }"
) as demo:
    gr.Markdown(
        """# 🎬 Temporal Sync Inspector
        **Powered by Gemma 4 12B — running fully local via Ollama**

        Detects audio-visual desync in video. Unlike traditional pipelines that run a separate
        ASR model + vision encoder, Gemma 4 12B processes raw audio frames and image patches
        through the **same backbone with shared RoPE positional embeddings** — so it can
        natively reason about time alignment between what it sees and what it hears.

        > Works offline. No API keys. No cloud. 16GB RAM.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload video", sources=["upload"])

            gr.Markdown("### Quick check (single window)")
            with gr.Row():
                window_start = gr.Number(label="Start (seconds)", value=0, minimum=0)
                window_dur_quick = gr.Number(label="Duration (seconds)", value=30, minimum=5, maximum=60)
            quick_btn = gr.Button("Analyze window", variant="primary")

            gr.Markdown("### Full video analysis")
            with gr.Row():
                window_dur_full = gr.Number(label="Window size (s)", value=30, minimum=10, maximum=60)
                fps_input = gr.Slider(label="Frames per second", minimum=1, maximum=2, step=1, value=1)
            full_btn = gr.Button("Analyze full video", variant="secondary")

            gr.Markdown(
                "_Tip: 1 FPS is enough for lip sync detection. "
                "Use 2 FPS for fast-motion content._"
            )

        with gr.Column(scale=2):
            result_md = gr.Markdown(label="Result", value="_Results will appear here…_")
            with gr.Accordion("Raw JSON", open=False):
                result_json = gr.Code(language="json", label="Raw model output")

    quick_btn.click(
        run_quick_check,
        inputs=[video_input, window_start, window_dur_quick],
        outputs=[result_md, result_json],
    )

    full_btn.click(
        run_full_analysis,
        inputs=[video_input, window_dur_full, fps_input],
        outputs=[result_md, result_json],
    )

    gr.Markdown(
        """---
        **How it works:** Frames are extracted at 1 FPS and audio at 16kHz (Gemma's native rate).
        Both are sent in a single Ollama chat call — no intermediate transcription, no separate encoder.
        The model's shared positional space lets it correlate mouth shapes with phonemes at the token level.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
