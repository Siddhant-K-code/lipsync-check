# temporal-sync-inspector

Detects audio-visual desync in video files using Gemma 4's encoder-free multimodal architecture.

Audio and video share the same RoPE positional space — the model reasons about time alignment between modalities in a single forward pass. No Whisper, no separate vision encoder, no cloud.

## How it works

Traditional pipelines run a vision encoder + ASR model separately, then ask an LLM to correlate two independent representations. Gemma 4 projects raw 16kHz audio frames and image patches through the **same backbone** with shared positional embeddings — so it can natively correlate mouth shapes with phonemes at the token level.

## Install

### Prerequisites

```bash
# ffmpeg + ffprobe (required for frame/audio extraction)
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu/Debian

# Ollama with Gemma 4 E4B
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e4b       # 9.6GB, 8GB RAM, audio + vision
```

### Build from source

```bash
git clone https://github.com/Siddhant-K-code/temporal-sync-inspector
cd temporal-sync-inspector
git checkout go-rewrite
go build -o tsi ./cmd/tsi
```

### Install to PATH

```bash
go install github.com/Siddhant-K-code/temporal-sync-inspector/cmd/tsi@latest
```

## Usage

```bash
# Analyze full video (sliding 30s windows)
tsi video.mp4

# Quick check — single window
tsi video.mp4 --quick --start 0 --duration 30

# Custom window size and FPS
tsi video.mp4 --window 60 --fps 2

# JSON output (pipe to jq, scripts, etc.)
tsi video.mp4 --json | jq '.verdict'

# Remote Ollama server
tsi video.mp4 --host http://192.168.1.10:11434

# Smaller/faster model
tsi video.mp4 --model gemma4:e2b
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4:e4b` | Ollama model |
| `--host` | `http://localhost:11434` | Ollama host |
| `--fps` | `1` | Frames per second to extract |
| `--window` | `30` | Analysis window size (seconds) |
| `--start` | `0` | Start time for `--quick` mode |
| `--duration` | `30` | Duration for `--quick` mode |
| `--quick` | false | Analyze a single window only |
| `--json` | false | Output raw JSON |

## Output

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

## Project structure

```
temporal-sync-inspector/
├── cmd/tsi/main.go              # CLI entry point (cobra)
├── internal/
│   ├── extract/extract.go       # ffmpeg wrapper — frames + 16kHz audio
│   ├── ollama/client.go         # Ollama /api/chat multimodal client
│   └── inspector/inspector.go   # windowed analysis + summary
├── go.mod
└── README.md
```

## License

MIT
