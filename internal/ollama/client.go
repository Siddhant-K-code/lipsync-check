// Package ollama provides a minimal client for Ollama's /api/chat endpoint.
// Sends multimodal messages (images + audio as base64) in a single request —
// exploiting Gemma 4's encoder-free architecture where audio and image tokens
// share the same RoPE positional space as text.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const defaultHost = "http://localhost:11434"

// Client is a thin Ollama API client.
type Client struct {
	Host       string
	Model      string
	HTTPClient *http.Client
}

// New returns a Client with sensible defaults.
func New(host, model string) *Client {
	if host == "" {
		host = defaultHost
	}
	return &Client{
		Host:  host,
		Model: model,
		HTTPClient: &http.Client{
			Timeout: 180 * time.Second,
		},
	}
}

// message is the Ollama chat message format.
type message struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"` // base64-encoded images or audio
}

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
	Stream   bool      `json:"stream"`
	Options  map[string]any `json:"options,omitempty"`
}

type chatResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	Error string `json:"error,omitempty"`
}

// Chat sends a multimodal prompt and returns the model's text response.
// images accepts base64-encoded JPEG frames and/or WAV audio — Ollama
// passes all of them through Gemma 4's unified token projection.
func (c *Client) Chat(ctx context.Context, system, prompt string, images []string) (string, error) {
	msgs := []message{}
	if system != "" {
		msgs = append(msgs, message{Role: "system", Content: system})
	}
	msgs = append(msgs, message{
		Role:    "user",
		Content: prompt,
		Images:  images,
	})

	body := chatRequest{
		Model:    c.Model,
		Messages: msgs,
		Stream:   false,
		Options:  map[string]any{"temperature": 0.0},
	}

	data, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.Host+"/api/chat", bytes.NewReader(data))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, raw)
	}

	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return "", fmt.Errorf("parse response: %w\nraw: %s", err, raw)
	}
	if cr.Error != "" {
		return "", fmt.Errorf("ollama error: %s", cr.Error)
	}
	return cr.Message.Content, nil
}

// Ping checks that Ollama is reachable and the model is available.
func (c *Client) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.Host+"/api/tags", nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("cannot reach Ollama at %s: %w", c.Host, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama /api/tags returned %d", resp.StatusCode)
	}
	return nil
}
