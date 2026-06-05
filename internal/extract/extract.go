// Package extract wraps ffmpeg to pull frames and audio from a video file.
// Frames are extracted at 1 FPS (configurable), scaled to 640px wide.
// Audio is extracted as 16kHz mono WAV — Gemma 4's native input rate.
package extract

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Frame holds a single video frame with its timestamp.
type Frame struct {
	TimestampS float64
	Base64JPEG string
	Path       string
}

// ExtractFrames pulls frames from video at the given FPS into a temp dir.
// Returns at most maxFrames frames starting from startS.
func ExtractFrames(videoPath string, startS, durationS float64, fps, maxFrames int) ([]Frame, error) {
	dir, err := os.MkdirTemp("", "tsi_frames_*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}

	args := []string{
		"-ss", fmt.Sprintf("%.3f", startS),
		"-i", videoPath,
		"-t", fmt.Sprintf("%.3f", durationS),
		"-vf", fmt.Sprintf("fps=%d,scale=640:-1", fps),
		"-q:v", "3",
		filepath.Join(dir, "frame_%04d.jpg"),
		"-y", "-loglevel", "error",
	}

	if out, err := exec.Command("ffmpeg", args...).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("ffmpeg frames: %w\n%s", err, out)
	}

	entries, err := filepath.Glob(filepath.Join(dir, "frame_*.jpg"))
	if err != nil {
		return nil, err
	}
	sort.Strings(entries)

	if maxFrames > 0 && len(entries) > maxFrames {
		entries = entries[:maxFrames]
	}

	frames := make([]Frame, 0, len(entries))
	for i, path := range entries {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		frames = append(frames, Frame{
			TimestampS: startS + float64(i)/float64(fps),
			Base64JPEG: base64.StdEncoding.EncodeToString(data),
			Path:       path,
		})
	}
	return frames, nil
}

// ExtractAudio pulls a WAV segment (16kHz mono) from the video.
// Returns the path to the temp WAV file.
func ExtractAudio(videoPath string, startS, durationS float64) (string, error) {
	f, err := os.CreateTemp("", "tsi_audio_*.wav")
	if err != nil {
		return "", fmt.Errorf("create temp wav: %w", err)
	}
	f.Close()

	args := []string{
		"-ss", fmt.Sprintf("%.3f", startS),
		"-i", videoPath,
		"-t", fmt.Sprintf("%.3f", durationS),
		"-ar", "16000", // 16kHz — Gemma 4 native rate
		"-ac", "1",     // mono
		"-f", "wav",
		f.Name(),
		"-y", "-loglevel", "error",
	}

	if out, err := exec.Command("ffmpeg", args...).CombinedOutput(); err != nil {
		return "", fmt.Errorf("ffmpeg audio: %w\n%s", err, out)
	}
	return f.Name(), nil
}

// AudioToBase64 reads a WAV file and returns its base64 encoding.
func AudioToBase64(wavPath string) (string, error) {
	data, err := os.ReadFile(wavPath)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

// Duration returns the video duration in seconds via ffprobe.
func Duration(videoPath string) (float64, error) {
	out, err := exec.Command(
		"ffprobe", "-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath,
	).Output()
	if err != nil {
		return 0, fmt.Errorf("ffprobe: %w", err)
	}
	s := strings.TrimSpace(string(out))
	d, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration %q: %w", s, err)
	}
	return d, nil
}

// Cleanup removes temp files created during extraction.
func Cleanup(paths ...string) {
	for _, p := range paths {
		os.Remove(p)
		// Also try removing parent dir if it looks like our temp dir
		dir := filepath.Dir(p)
		if strings.Contains(filepath.Base(dir), "tsi_") {
			os.RemoveAll(dir)
		}
	}
}
