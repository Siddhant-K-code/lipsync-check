# Temporal Sync Inspector

Detects audio-visual desync in video files using Gemma 4 12B's encoder-free architecture.
Audio and video share the same RoPE positional space — so the model reasons about time
alignment between modalities in a single forward pass. No Whisper, no separate vision encoder.

## What it does

- Extracts frames + audio from a video
- Feeds both into Gemma 4 12B via Ollama in one prompt
- Reports: sync status, desync timestamps, confidence, and reasoning
- Works fully offline on 16GB RAM

## Setup

```bash
# 1. Pull the model
ollama pull gemma4:12b

# 2. Install deps
pip install -r requirements.txt

# 3. Run
python app.py
```

## How it works

Traditional pipelines: `video → vision encoder → embeddings` + `audio → ASR → text` → LLM reasons over two separate representations.

Gemma 4 12B: raw audio frames (16kHz, 40ms chunks) and image patches both project directly into the LLM's token space. RoPE handles temporal ordering for both. The model can natively correlate "at t=2.3s, the mouth shape doesn't match the phoneme in the audio."
