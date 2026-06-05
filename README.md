# temporal-sync-inspector

Detects audio-visual desync in video files using Gemma 4's encoder-free multimodal architecture. Audio and video share the same RoPE positional space — the model reasons about time alignment in a single forward pass. No Whisper, no separate vision encoder, no cloud.

Two implementations available:
- **`main` branch** — Python + Gradio (web UI)
- **`go-rewrite` branch** — Go CLI (single binary, no Python needed)

---

## Prerequisites

### 1. Ollama + Gemma 4 E4B

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh   # macOS/Linux
# Windows: https://ollama.com/download

# Pull the model (9.6GB)
ollama pull gemma4:e4b

# Verify
ollama list
# should show: gemma4:e4b
```

### 2. ffmpeg (required for both versions)

```bash
brew install ffmpeg          # macOS (Homebrew)
sudo apt install ffmpeg      # Ubuntu/Debian
sudo dnf install ffmpeg      # Fedora
# Windows: https://ffmpeg.org/download.html — add ffmpeg/ffprobe to PATH
```

**macOS note:** If you don't have Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

---

## Go CLI (go-rewrite branch) — recommended

Single binary, no Python, no pip.

### Install

```bash
# Option A — build from source
git clone https://github.com/Siddhant-K-code/temporal-sync-inspector
cd temporal-sync-inspector
git checkout go-rewrite
go build -o tsi ./cmd/tsi

# Option B — install directly
go install github.com/Siddhant-K-code/temporal-sync-inspector/cmd/tsi@latest
```

### Run

```bash
# Analyze full video (slides a 30s window across the whole file)
tsi video.mp4

# Quick check — single window starting at 0s
tsi video.mp4 --quick --start 0 --duration 30

# Check a specific section (e.g. t=60s for 45s)
tsi video.mp4 --quick --start 60 --duration 45

# Higher frame rate for fast-motion content
tsi video.mp4 --fps 2

# JSON output — pipe to jq or scripts
tsi video.mp4 --json
tsi video.mp4 --json | jq '.verdict'
tsi video.mp4 --json | jq '.windows[] | select(.in_sync == false)'

# Use a lighter model if RAM is tight
tsi video.mp4 --model gemma4:e2b

# Remote Ollama server
tsi video.mp4 --host http://192.168.1.10:11434
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4:e4b` | Ollama model |
| `--host` | `http://localhost:11434` | Ollama host |
| `--fps` | `1` | Frames per second to extract (1–2) |
| `--window` | `30` | Window size in seconds (full mode) |
| `--quick` | false | Analyze a single window only |
| `--start` | `0` | Start time in seconds (quick mode) |
| `--duration` | `30` | Duration in seconds (quick mode) |
| `--json` | false | Output raw JSON instead of formatted report |

---

## Python + Gradio UI (main branch)

### Install

```bash
git clone https://github.com/Siddhant-K-code/temporal-sync-inspector
cd temporal-sync-inspector
# main branch is already checked out
pip install -r requirements.txt
```

### Run

```bash
python app.py
# Opens at http://localhost:7860
```

Upload a video, choose quick check or full analysis, click Analyze.

---

## Example output (Go CLI)

```
  Temporal Sync Inspector
  Model: gemma4:e4b  |  Window: 30s  |  FPS: 1

  ✗ SIGNIFICANT DESYNC
  Sync score:  33.3%
  Windows:     3 analyzed, 2 with desync
  Desync at:   12.4s, 67.1s

  Per-window breakdown
  ────────────────────────────────────────────────────────────
  ✓ t=0s–30s   [high confidence]
    Audio and video appear synchronized throughout.

  ✗ t=30s–60s  [high confidence]  [lip_sync]  (~180ms offset)
    Mouth movements lag behind audio by approximately 180ms.

  ✗ t=60s–90s  [medium confidence]  [action_sync]
    Clapping sounds precede visible hand contact by ~2 frames.
```

## JSON schema

```json
{
  "total_windows": 3,
  "desync_windows": 2,
  "sync_score": 33.3,
  "verdict": "SIGNIFICANT DESYNC",
  "desync_timestamps": [12.4, 67.1],
  "windows": [
    {
      "window_start_s": 0,
      "window_end_s": 30,
      "in_sync": true,
      "confidence": "high",
      "desync_detected_at_s": null,
      "desync_type": null,
      "estimated_offset_ms": null,
      "reasoning": "Audio and video appear synchronized.",
      "suspicious_frames": []
    }
  ]
}
```

---

## How it works

Traditional pipelines: `video → vision encoder → embeddings` + `audio → Whisper → text` → LLM correlates two separate representations.

Gemma 4 E4B: raw 16kHz audio frames and image patches both project directly into the LLM's token space via the same linear projection. RoPE handles temporal ordering for both modalities. The model can natively say "at t=2.3s, the mouth shape doesn't match the phoneme in the audio" — in a single forward pass.

---

## Project structure

```
temporal-sync-inspector/
├── cmd/tsi/main.go              # Go CLI (cobra) — go-rewrite branch
├── internal/
│   ├── extract/extract.go       # ffmpeg wrapper — frames + 16kHz WAV
│   ├── ollama/client.go         # Ollama /api/chat multimodal client
│   └── inspector/inspector.go   # windowed analysis engine + summary
├── app.py                       # Python Gradio UI — main branch
├── extract.py                   # Python ffmpeg wrapper
├── inspector.py                 # Python analysis logic
└── requirements.txt
```

---

## License

MIT
