# temporal-sync-inspector

Detects audio-visual desync in video files using [Gemma 4](https://ollama.com/library/gemma4) via Ollama.

```bash
tsi video.mp4
tsi video.mp4 --quick --start 60 --duration 30
tsi video.mp4 --json | jq '.verdict'
```

No Python. No cloud. Single binary.

## Requirements

- [Ollama](https://ollama.com) running locally with `gemma4:e4b` pulled
- `ffmpeg` on your PATH

```bash
ollama pull gemma4:e4b
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu/Debian
```

## Install

```bash
# Build from source
git clone https://github.com/Siddhant-K-code/temporal-sync-inspector
cd temporal-sync-inspector
go build -o tsi ./cmd/tsi

# Or install directly
go install github.com/Siddhant-K-code/temporal-sync-inspector/cmd/tsi@latest
```

## Usage

```bash
tsi video.mp4                          # full analysis (sliding 30s windows)
tsi video.mp4 --quick --start 0       # single window from t=0s
tsi video.mp4 --quick --start 60 --duration 45
tsi video.mp4 --fps 2                  # higher frame rate for fast motion
tsi video.mp4 --json                   # raw JSON output
tsi video.mp4 --model gemma4:e4b      # specify model
tsi video.mp4 --host http://192.168.1.10:11434  # remote Ollama
```

<details>
<summary>All flags</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4:e4b` | Ollama model |
| `--host` | `http://localhost:11434` | Ollama host |
| `--fps` | `1` | Frames per second to extract |
| `--window` | `30` | Window size in seconds |
| `--quick` | false | Analyze a single window only |
| `--start` | `0` | Start time in seconds (quick mode) |
| `--duration` | `30` | Duration in seconds (quick mode) |
| `--json` | false | Output raw JSON |

</details>

## Example output

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
    Audio and video appear synchronized.

  ✗ t=30s–60s  [high confidence]  [lip_sync]  (~180ms offset)
    Mouth movements lag behind audio by ~180ms.

  ✗ t=60s–90s  [medium confidence]  [action_sync]
    Clapping sounds precede visible hand contact by ~2 frames.
```

<details>
<summary>JSON schema</summary>

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

</details>

## Project structure

```
temporal-sync-inspector/
├── cmd/tsi/main.go
└── internal/
    ├── extract/extract.go      # ffmpeg wrapper — frames + 16kHz WAV
    ├── ollama/client.go        # Ollama /api/chat multimodal client
    └── inspector/inspector.go  # windowed analysis + summary
```

## License

MIT
