// Package inspector drives the audio-visual sync analysis.
// It slides a window across the video, sends frames + audio to Gemma 4
// in a single multimodal prompt, and parses the structured JSON result.
package inspector

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Siddhant-K-code/lipsync-check/internal/extract"
	"github.com/Siddhant-K-code/lipsync-check/internal/ollama"
)

const systemPrompt = `You are an audio-visual sync analyzer. You receive video frames (with timestamps) and the corresponding audio from the same time window.

Your job: determine whether the audio and video are synchronized.

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
}`

// WindowResult holds the analysis result for a single time window.
type WindowResult struct {
	WindowStartS float64  `json:"window_start_s"`
	WindowEndS   float64  `json:"window_end_s"`
	InSync       *bool    `json:"in_sync"`
	Confidence   string   `json:"confidence"`
	DesyncAtS    *float64 `json:"desync_detected_at_s"`
	DesyncType   *string  `json:"desync_type"`
	OffsetMS     *float64 `json:"estimated_offset_ms"`
	Reasoning    string   `json:"reasoning"`
	SuspFrames   []float64 `json:"suspicious_frames"`
	ParseError   bool     `json:"parse_error,omitempty"`
	RawResponse  string   `json:"raw_response,omitempty"`
}

// Summary aggregates all window results into a video-level verdict.
type Summary struct {
	TotalWindows    int             `json:"total_windows"`
	DesyncWindows   int             `json:"desync_windows"`
	SyncScore       float64         `json:"sync_score"`
	Verdict         string          `json:"verdict"`
	DesyncTimestamps []float64      `json:"desync_timestamps"`
	Windows         []WindowResult  `json:"windows"`
}

// Config holds inspector parameters.
type Config struct {
	OllamaHost string
	Model      string
	FPS        int
	WindowS    float64
	MaxFrames  int
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		OllamaHost: "http://localhost:11434",
		Model:      "gemma4:e4b",
		FPS:        1,
		WindowS:    30.0,
		MaxFrames:  60,
	}
}

// ProgressFunc is called after each window with (windowIndex, totalWindows, startS).
type ProgressFunc func(idx, total int, startS float64)

// InspectVideo slides a window across the full video and inspects each segment.
func InspectVideo(ctx context.Context, videoPath string, cfg Config, progress ProgressFunc) (*Summary, error) {
	client := ollama.New(cfg.OllamaHost, cfg.Model)

	if err := client.Ping(ctx); err != nil {
		return nil, fmt.Errorf("ollama not reachable: %w", err)
	}

	duration, err := extract.Duration(videoPath)
	if err != nil {
		return nil, fmt.Errorf("get duration: %w", err)
	}

	var results []WindowResult
	total := int(duration/cfg.WindowS) + 1
	idx := 0

	for start := 0.0; start < duration; start += cfg.WindowS {
		windowDur := cfg.WindowS
		if start+windowDur > duration {
			windowDur = duration - start
		}
		if windowDur < 5 {
			break
		}

		if progress != nil {
			progress(idx, total, start)
		}

		result, err := inspectWindow(ctx, client, videoPath, start, windowDur, cfg)
		if err != nil {
			// Non-fatal: record the error and continue
			t := true
			result = WindowResult{
				WindowStartS: start,
				WindowEndS:   start + windowDur,
				InSync:       &t,
				Confidence:   "low",
				Reasoning:    fmt.Sprintf("analysis failed: %v", err),
				ParseError:   true,
			}
		}
		results = append(results, result)
		idx++
	}

	return summarize(results), nil
}

// InspectWindow analyzes a single time window.
func InspectWindow(ctx context.Context, videoPath string, startS, durationS float64, cfg Config) (*WindowResult, error) {
	client := ollama.New(cfg.OllamaHost, cfg.Model)
	r, err := inspectWindow(ctx, client, videoPath, startS, durationS, cfg)
	if err != nil {
		return nil, err
	}
	return &r, nil
}

func inspectWindow(ctx context.Context, client *ollama.Client, videoPath string, startS, durationS float64, cfg Config) (WindowResult, error) {
	result := WindowResult{
		WindowStartS: startS,
		WindowEndS:   startS + durationS,
	}

	// Extract frames
	frames, err := extract.ExtractFrames(videoPath, startS, durationS, cfg.FPS, cfg.MaxFrames)
	if err != nil {
		return result, fmt.Errorf("extract frames: %w", err)
	}
	defer func() {
		for _, f := range frames {
			extract.Cleanup(f.Path)
		}
	}()

	// Extract audio (16kHz WAV — Gemma 4 native)
	wavPath, err := extract.ExtractAudio(videoPath, startS, durationS)
	if err != nil {
		return result, fmt.Errorf("extract audio: %w", err)
	}
	defer extract.Cleanup(wavPath)

	audioB64, err := extract.AudioToBase64(wavPath)
	if err != nil {
		return result, fmt.Errorf("encode audio: %w", err)
	}

	// Build image list: frames + audio in one slice
	// Gemma 4's encoder-free design handles both through the same token projection
	images := make([]string, 0, len(frames)+1)
	for _, f := range frames {
		images = append(images, f.Base64JPEG)
	}
	images = append(images, audioB64)

	// Build prompt
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Analyze this %.0fs video segment (t=%.1fs to t=%.1fs).\n\nFrames provided:\n",
		durationS, startS, startS+durationS))
	for _, f := range frames {
		sb.WriteString(fmt.Sprintf("  - Frame at t=%.1fs\n", f.TimestampS))
	}
	sb.WriteString("\nThe audio for this exact window is also provided. Determine if audio and video are in sync.")

	raw, err := client.Chat(ctx, systemPrompt, sb.String(), images)
	if err != nil {
		return result, fmt.Errorf("ollama chat: %w", err)
	}

	// Strip markdown fences if present
	cleaned := strings.TrimSpace(raw)
	if strings.HasPrefix(cleaned, "```") {
		parts := strings.SplitN(cleaned, "\n", 2)
		if len(parts) == 2 {
			cleaned = parts[1]
		}
		cleaned = strings.TrimSuffix(strings.TrimSpace(cleaned), "```")
	}

	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		result.ParseError = true
		result.RawResponse = raw[:min(len(raw), 400)]
		result.Reasoning = fmt.Sprintf("parse error: %v", err)
	}

	result.WindowStartS = startS
	result.WindowEndS = startS + durationS
	return result, nil
}

func summarize(results []WindowResult) *Summary {
	var desyncWindows int
	var desyncTS []float64

	for _, r := range results {
		if r.InSync != nil && !*r.InSync {
			desyncWindows++
			if r.DesyncAtS != nil {
				desyncTS = append(desyncTS, *r.DesyncAtS)
			}
		}
	}

	total := len(results)
	score := 0.0
	if total > 0 {
		score = float64(total-desyncWindows) / float64(total) * 100
	}

	verdict := "IN SYNC"
	if desyncWindows > 0 {
		ratio := float64(desyncWindows) / float64(max(total, 1))
		if ratio >= 0.3 {
			verdict = "SIGNIFICANT DESYNC"
		} else {
			verdict = "MINOR DESYNC"
		}
	}

	return &Summary{
		TotalWindows:     total,
		DesyncWindows:    desyncWindows,
		SyncScore:        score,
		Verdict:          verdict,
		DesyncTimestamps: desyncTS,
		Windows:          results,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
