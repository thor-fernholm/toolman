package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
)

func main() {
	// Load env once at startup.
	_ = godotenv.Load()
	_ = godotenv.Load("../../.env")

	server, err := newServerFromEnv()
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	// Lists available generation models from the Bellman backend.
	mux.HandleFunc("/models", server.handleModels)
	// Main judge endpoint.
	mux.HandleFunc("/judge", server.handleJudge)
	// Backwards compatible path.
	mux.HandleFunc("/toolman", server.handleJudge)

	addr := server.host + ":" + strconv.Itoa(server.port)
	fmt.Println("Listening on", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

type JudgeRequest struct {
	QueryID string `json:"query_id,omitempty"`
	Query   string `json:"query"`
	Answer  string `json:"answer"`
}

type JudgeResult struct {
	AnswerStatus string `json:"answer_status"`
	Reason       string `json:"reason"`
}

type JudgeResponse struct {
	QueryID      string      `json:"query_id,omitempty"`
	Model        string      `json:"model"`
	DurationMs   int64       `json:"duration_ms"`
	Result       JudgeResult `json:"result"`
	RawText      string      `json:"raw_text,omitempty"`
	ErrorMessage string      `json:"error,omitempty"`
}

type server struct {
	client *bellman.Bellman
	model  gen.Model
	host   string
	port   int
}

func newServerFromEnv() (*server, error) {
	bellmanURL := strings.TrimSpace(os.Getenv("BELLMAN_URL"))
	bellmanToken := strings.TrimSpace(os.Getenv("BELLMAN_TOKEN"))
	if bellmanURL == "" || bellmanToken == "" {
		return nil, errors.New("missing BELLMAN_URL/BELLMAN_TOKEN in environment")
	}

	modelFQN := strings.TrimSpace(os.Getenv("EVAL_MODEL"))
	if modelFQN == "" {
		modelFQN = strings.TrimSpace(os.Getenv("JUDGE_MODEL"))
	}
	if modelFQN == "" {
		return nil, errors.New("missing EVAL_MODEL (or JUDGE_MODEL) in environment. Hint: call GET /models to see what your Bellman backend supports")
	}

	m, err := gen.ToModel(modelFQN)
	if err != nil {
		return nil, fmt.Errorf("invalid EVAL_MODEL/JUDGE_MODEL: %w", err)
	}

	host := strings.TrimSpace(os.Getenv("BELL_JUDGE_HOST"))
	if host == "" {
		host = "localhost"
	}
	port := 8080
	if p := strings.TrimSpace(os.Getenv("BELL_JUDGE_PORT")); p != "" {
		pi, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid BELL_JUDGE_PORT: %w", err)
		}
		port = pi
	}

	return &server{
		client: bellman.New(bellmanURL, bellman.Key{Name: "judge", Token: bellmanToken}),
		model:  m,
		host:   host,
		port:   port,
	}, nil
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	models, err := s.client.GenModels()
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, models)
}

func (s *server) handleJudge(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	// Avoid unbounded request bodies.
	r.Body = http.MaxBytesReader(w, r.Body, 5<<20) // 5MB
	defer r.Body.Close()

	var req JudgeRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, JudgeResponse{ErrorMessage: "invalid json: " + err.Error()})
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeJSON(w, http.StatusBadRequest, JudgeResponse{ErrorMessage: "missing 'query'"})
		return
	}
	if strings.TrimSpace(req.Answer) == "" {
		writeJSON(w, http.StatusBadRequest, JudgeResponse{QueryID: req.QueryID, ErrorMessage: "missing 'answer'"})
		return
	}

	start := time.Now()
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	system := strings.TrimSpace(os.Getenv("JUDGE_SYSTEM"))
	if system == "" {
		system = "You are a strict but fair evaluator. Judge whether the answer satisfies the user's query. Focus on completeness and relevance. Do not be overly harsh. Output must be JSON matching the provided schema."
	}

	userPrompt := buildJudgePrompt(req.Query, req.Answer)
	outputSchema := &schema.JSON{
		Type: schema.Object,
		Properties: map[string]*schema.JSON{
			"answer_status": {
				Type:        schema.String,
				Description: "Either 'Solved' or 'Unsolved'.",
				Enum:        []any{"Solved", "Unsolved"},
			},
			"reason": {
				Type:        schema.String,
				Description: "Short justification.",
			},
		},
		Required: []string{"answer_status", "reason"},
	}

	llm := s.client.Generator().
		Model(s.model).
		System(system).
		Temperature(0).
		StrictOutput(true).
		Output(outputSchema).
		WithContext(ctx)

	resp, err := llm.Prompt(prompt.AsUser(userPrompt))
	if err != nil {
		writeJSON(w, http.StatusBadGateway, JudgeResponse{QueryID: req.QueryID, Model: s.model.FQN(), DurationMs: time.Since(start).Milliseconds(), ErrorMessage: err.Error()})
		return
	}

	raw, _ := resp.AsText()
	var out JudgeResult
	if err := resp.Unmarshal(&out); err != nil {
		// Provide raw text for debugging.
		writeJSON(w, http.StatusBadGateway, JudgeResponse{QueryID: req.QueryID, Model: s.model.FQN(), DurationMs: time.Since(start).Milliseconds(), RawText: raw, ErrorMessage: "failed to parse model output as json: " + err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, JudgeResponse{
		QueryID:    req.QueryID,
		Model:      s.model.FQN(),
		DurationMs: time.Since(start).Milliseconds(),
		Result:     out,
	})
}

func buildJudgePrompt(query, answer string) string {
	// Keep prompt short and deterministic.
	var b strings.Builder
	b.WriteString("Decide if the answer solves the query.\n")
	b.WriteString("Rules:\n")
	b.WriteString("- Return Solved if the answer makes a genuine attempt to address ALL parts of the query.\n")
	b.WriteString("- Return Unsolved if it refuses, is unrelated, or misses one or more major parts.\n")
	b.WriteString("- Assume facts are correct unless there is a severe and obvious error.\n")
	b.WriteString("\nQuery:\n")
	b.WriteString(query)
	b.WriteString("\n\nAnswer:\n")
	b.WriteString(answer)
	return b.String()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// Unused but handy for debugging large payloads.
func readAll(r io.Reader) string {
	b, _ := io.ReadAll(r)
	return string(b)
}
