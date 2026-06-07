package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/fatih/color"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"

	"github.com/Siddhant-K-code/lipsync-check/internal/inspector"
)

// Set by goreleaser via ldflags at build time.
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

var (
	flagModel      string
	flagHost       string
	flagFPS        int
	flagWindowS    float64
	flagStartS     float64
	flagDurationS  float64
	flagJSON       bool
	flagQuick      bool
)

var bold   = color.New(color.Bold)
var green  = color.New(color.FgGreen, color.Bold)
var yellow = color.New(color.FgYellow, color.Bold)
var red    = color.New(color.FgRed, color.Bold)
var dim    = color.New(color.Faint)
var cyan   = color.New(color.FgCyan)

var rootCmd = &cobra.Command{
	Use:   "lipsync-check <video>",
	Short: "Temporal Sync Inspector — detect A/V desync using Gemma 4",
	Long: `Temporal Sync Inspector

Detects audio-visual desync in video files using Gemma 4's encoder-free
multimodal architecture. Audio and video share the same RoPE positional
space — the model reasons about time alignment in a single forward pass.

No Whisper. No separate vision encoder. Fully local via Ollama.

Examples:
  lipsync-check video.mp4
  lipsync-check video.mp4 --quick --start 10 --duration 30
  lipsync-check video.mp4 --window 60 --fps 2 --json
  lipsync-check video.mp4 --model gemma4:e2b --host http://192.168.1.10:11434`,
	Args: cobra.ExactArgs(1),
	RunE: run,
}

func init() {
	rootCmd.Version = fmt.Sprintf("%s (commit %s, built %s)", version, commit, date)
	rootCmd.Flags().StringVar(&flagModel, "model", "gemma4:e4b", "Ollama model to use")
	rootCmd.Flags().StringVar(&flagHost, "host", "http://localhost:11434", "Ollama host URL")
	rootCmd.Flags().IntVar(&flagFPS, "fps", 1, "Frames per second to extract (1–2)")
	rootCmd.Flags().Float64Var(&flagWindowS, "window", 30, "Analysis window size in seconds")
	rootCmd.Flags().Float64Var(&flagStartS, "start", 0, "Start time in seconds (quick mode)")
	rootCmd.Flags().Float64Var(&flagDurationS, "duration", 30, "Duration in seconds (quick mode)")
	rootCmd.Flags().BoolVar(&flagJSON, "json", false, "Output raw JSON instead of formatted report")
	rootCmd.Flags().BoolVar(&flagQuick, "quick", false, "Analyze a single window instead of full video")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func run(cmd *cobra.Command, args []string) error {
	videoPath := args[0]

	if _, err := os.Stat(videoPath); err != nil {
		return fmt.Errorf("video file not found: %s", videoPath)
	}

	cfg := inspector.Config{
		OllamaHost: flagHost,
		Model:      flagModel,
		FPS:        flagFPS,
		WindowS:    flagWindowS,
		MaxFrames:  int(flagWindowS) * flagFPS,
	}

	ctx := context.Background()

	if flagQuick {
		return runQuick(ctx, videoPath, cfg)
	}
	return runFull(ctx, videoPath, cfg)
}

func runQuick(ctx context.Context, videoPath string, cfg inspector.Config) error {
	if !flagJSON {
		bold.Printf("\n  Temporal Sync Inspector\n")
		dim.Printf("  Model: %s  |  Quick mode: t=%.0fs–%.0fs\n\n",
			cfg.Model, flagStartS, flagStartS+flagDurationS)
	}

	result, err := inspector.InspectWindow(ctx, videoPath, flagStartS, flagDurationS, cfg)
	if err != nil {
		return err
	}

	if flagJSON {
		return printJSON(result)
	}

	printWindowResult(*result)
	return nil
}

func runFull(ctx context.Context, videoPath string, cfg inspector.Config) error {
	if !flagJSON {
		bold.Printf("\n  Temporal Sync Inspector\n")
		dim.Printf("  Model: %s  |  Window: %.0fs  |  FPS: %d\n\n",
			cfg.Model, cfg.WindowS, cfg.FPS)
	}

	var bar *progressbar.ProgressBar
	if !flagJSON {
		bar = progressbar.NewOptions(-1,
			progressbar.OptionSetDescription("  Analyzing"),
			progressbar.OptionSetTheme(progressbar.Theme{
				Saucer:        "█",
				SaucerPadding: "░",
				BarStart:      "  [",
				BarEnd:        "]",
			}),
			progressbar.OptionShowCount(),
			progressbar.OptionClearOnFinish(),
		)
	}

	summary, err := inspector.InspectVideo(ctx, videoPath, cfg, func(idx, total int, startS float64) {
		if bar != nil {
			bar.Describe(fmt.Sprintf("  Analyzing t=%.0fs", startS))
			bar.Add(1)
		}
	})
	if err != nil {
		return err
	}

	if bar != nil {
		bar.Finish()
	}

	if flagJSON {
		return printJSON(summary)
	}

	printSummary(summary)
	return nil
}

func printSummary(s *inspector.Summary) {
	// Verdict banner
	fmt.Println()
	switch s.Verdict {
	case "IN SYNC":
		green.Printf("  ✓ %s\n", s.Verdict)
	case "MINOR DESYNC":
		yellow.Printf("  ⚠ %s\n", s.Verdict)
	default:
		red.Printf("  ✗ %s\n", s.Verdict)
	}

	fmt.Printf("  Sync score:  %.1f%%\n", s.SyncScore)
	fmt.Printf("  Windows:     %d analyzed, %d with desync\n", s.TotalWindows, s.DesyncWindows)

	if len(s.DesyncTimestamps) > 0 {
		ts := make([]string, len(s.DesyncTimestamps))
		for i, t := range s.DesyncTimestamps {
			ts[i] = fmt.Sprintf("%.1fs", t)
		}
		yellow.Printf("  Desync at:   %s\n", strings.Join(ts, ", "))
	}

	fmt.Println()
	bold.Println("  Per-window breakdown")
	fmt.Println("  " + strings.Repeat("─", 60))

	for _, w := range s.Windows {
		printWindowResult(w)
	}
}

func printWindowResult(w inspector.WindowResult) {
	icon := "✓"
	c := green
	inSync := w.InSync != nil && *w.InSync
	if !inSync {
		icon = "✗"
		c = red
	}
	if w.ParseError {
		icon = "?"
		c = yellow
	}

	c.Printf("  %s t=%.0fs–%.0fs", icon, w.WindowStartS, w.WindowEndS)

	conf := w.Confidence
	if conf == "" {
		conf = "?"
	}
	dim.Printf("  [%s confidence]", conf)

	if w.DesyncType != nil && *w.DesyncType != "" {
		cyan.Printf("  [%s]", *w.DesyncType)
	}
	if w.OffsetMS != nil {
		dim.Printf("  (~%.0fms offset)", *w.OffsetMS)
	}
	fmt.Println()

	if w.Reasoning != "" {
		dim.Printf("    %s\n", w.Reasoning)
	}
	fmt.Println()
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
