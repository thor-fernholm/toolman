package nestful

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/modfin/bellman"
	"github.com/modfin/bellman/agent"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

// PTCRunRequest is a server-side evaluation request. It is intended to be called by
// an external harness (e.g. NESTFUL) to run Bellman agent+tools in PTC mode.
type PTCRunRequest struct {
	TraceID string `json:"trace_id,omitempty"`

	// Model selects the backend model used by Bellman.
	// Example: {"provider":"openai","name":"gpt4o_mini"}
	Model gen.Model `json:"model"`

	// Query is the user input/question.
	Query string `json:"query"`

	// SystemPrompt is optional; it is prepended to Bellman's built-in PTC system fragment.
	SystemPrompt string `json:"system_prompt,omitempty"`

	// Tools is the per-sample NESTFUL tool specification list.
	Tools []NestfulToolSpec `json:"tools"`

	// PTC toggles whether tools should be adapted into code_execution.
	// If false, tools are exposed as normal tool-calls.
	UsePTC bool `json:"use_ptc"`

	PTCLanguage tools.ProgramLanguage `json:"ptc_language,omitempty"`

	MaxDepth    int `json:"max_depth,omitempty"`
	Parallelism int `json:"parallelism,omitempty"`

	Temperature *float64 `json:"temperature,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`

	// ExecutableFuncDir points to NESTFUL executable python functions directory.
	// Example: C:\\...\\NESTFUL-main\\data_v2\\executable_functions
	ExecutableFuncDir string `json:"executable_func_dir"`

	// PythonBin is the python executable to invoke. Default "python".
	PythonBin string `json:"python_bin,omitempty"`
}

type PTCRunResponse struct {
	Ok    bool   `json:"ok"`
	Error string `json:"error,omitempty"`

	TraceID string `json:"trace_id,omitempty"`

	FinalText string `json:"final_text,omitempty"`

	Metadata *RunMetadata `json:"metadata,omitempty"`

	// Prompts includes the outer conversation trace (user, tool-call, tool-response, assistant).
	// Useful to extract code_execution payload.
	Prompts []prompt.Prompt `json:"prompts,omitempty"`

	// ToolTrace is the executed underlying tool calls (inside PTC JS runtime).
	ToolTrace []ToolTraceEvent `json:"tool_trace,omitempty"`
}

type RunMetadata struct {
	Model        string `json:"model,omitempty"`
	InputTokens  int    `json:"input_tokens,omitempty"`
	OutputTokens int    `json:"output_tokens,omitempty"`
	TotalTokens  int    `json:"total_tokens,omitempty"`

	DurationMs int64 `json:"duration_ms,omitempty"`
}

type ToolTraceEvent struct {
	Index int `json:"index"`

	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`

	Ok          bool            `json:"ok"`
	Output      json.RawMessage `json:"output,omitempty"`
	OutputRaw   string          `json:"output_raw,omitempty"`
	Error       string          `json:"error,omitempty"`
	DurationMs  int64           `json:"duration_ms"`
	StartedAtMs int64           `json:"started_at_ms"`
}

// NestfulToolSpec matches the per-sample tool spec embedded in NESTFUL dataset.
type NestfulToolSpec struct {
	Name        string                  `json:"name"`
	Description string                  `json:"description"`
	Parameters  map[string]any          `json:"parameters"`
	Output      map[string]NestfulParam `json:"output_parameters"`
}

type NestfulParam struct {
	Type        any    `json:"type,omitempty"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
	Nullable    bool   `json:"nullable,omitempty"`
}

// PTCRunHandler returns an http.HandlerFunc implementing POST /ptc/run.
//
// NOTE: This file only provides the handler implementation. The bellmand server
// still needs to mount it on a router.
func PTCRunHandler(proxy *bellman.Proxy) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		var req PTCRunRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, PTCRunResponse{Ok: false, Error: "could not decode request: " + err.Error()})
			return
		}
		if strings.TrimSpace(req.Query) == "" {
			writeJSON(w, http.StatusBadRequest, PTCRunResponse{Ok: false, Error: "query is required"})
			return
		}
		if req.Model.Provider == "" || req.Model.Name == "" {
			writeJSON(w, http.StatusBadRequest, PTCRunResponse{Ok: false, Error: "model.provider and model.name are required"})
			return
		}
		if strings.TrimSpace(req.ExecutableFuncDir) == "" {
			writeJSON(w, http.StatusBadRequest, PTCRunResponse{Ok: false, Error: "executable_func_dir is required"})
			return
		}
		if req.MaxDepth <= 0 {
			req.MaxDepth = 10
		}
		if req.PTCLanguage == "" {
			req.PTCLanguage = tools.JavaScript
		}
		if req.PythonBin == "" {
			req.PythonBin = "python"
		}

		// Tool trace collector is captured in tool function closures.
		collector := &traceCollector{}

		bellmanTools := BuildTools(req.Tools, ToolRuntimeConfig{
			ExecutableFuncDir: req.ExecutableFuncDir,
			PythonBin:         req.PythonBin,
		}, collector, req.UsePTC)

		generator, err := proxy.Gen(req.Model)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, PTCRunResponse{Ok: false, Error: "could not get generator: " + err.Error(), TraceID: req.TraceID})
			return
		}
		generator = generator.System(req.SystemPrompt).
			SetTools(bellmanTools...).
			SetPTCLanguage(req.PTCLanguage)

		if req.Temperature != nil {
			generator = generator.Temperature(*req.Temperature)
		}
		if req.MaxTokens != nil {
			generator = generator.MaxTokens(*req.MaxTokens)
		}

		// Ensure request context is propagated through agent loop (outer tool calls).
		generator = generator.WithContext(r.Context())

		// Run agent loop.
		res, runErr := agent.Run[string](req.MaxDepth, req.Parallelism, generator, prompt.AsUser(req.Query))
		if runErr != nil {
			writeJSON(w, http.StatusOK, PTCRunResponse{
				Ok:      false,
				Error:   runErr.Error(),
				TraceID: req.TraceID,
				Metadata: &RunMetadata{
					Model:      req.Model.FQN(),
					DurationMs: time.Since(start).Milliseconds(),
				},
				Prompts:   resPromptsSafe(res),
				ToolTrace: collector.Events(),
			})
			return
		}

		writeJSON(w, http.StatusOK, PTCRunResponse{
			Ok:        true,
			TraceID:   req.TraceID,
			FinalText: res.Result,
			Metadata: &RunMetadata{
				Model:        res.Metadata.Model,
				InputTokens:  res.Metadata.InputTokens,
				OutputTokens: res.Metadata.OutputTokens,
				TotalTokens:  res.Metadata.TotalTokens,
				DurationMs:   time.Since(start).Milliseconds(),
			},
			Prompts:   res.Prompts,
			ToolTrace: collector.Events(),
		})
	}
}

func resPromptsSafe(res *agent.Result[string]) []prompt.Prompt {
	if res == nil {
		return nil
	}
	return res.Prompts
}

type ToolRuntimeConfig struct {
	ExecutableFuncDir string
	PythonBin         string
}

// BuildTools converts NESTFUL tool specs into Bellman tools.
// If enablePTC is true, the tools will be extracted and adapted into code_execution by Bellman.
func BuildTools(specs []NestfulToolSpec, cfg ToolRuntimeConfig, collector *traceCollector, enablePTC bool) []tools.Tool {
	out := make([]tools.Tool, 0, len(specs))
	for _, s := range specs {
		spec := s
		outKeys := sortedKeys(spec.Output)

		// Best-effort schema: since NESTFUL parameters are dynamic, we expose a generic object schema.
		// This keeps the tool available for execution; PTC docs will be less precise.
		argSchema := &schema.JSON{
			Type:                 schema.Object,
			Properties:           map[string]*schema.JSON{},
			AdditionalProperties: &schema.JSON{Type: schema.String},
		}

		t := tools.Tool{
			Name:           spec.Name,
			Description:    spec.Description,
			ArgumentSchema: argSchema,
			UsePTC:         enablePTC,
		}
		t.Function = func(ctx context.Context, call tools.Call) (string, error) {
			idx, startedAt := collector.start(spec.Name, call.Argument)
			toolStart := time.Now()

			outStr, err := execNestfulPython(ctx, cfg, spec.Name, call.Argument, outKeys)
			dur := time.Since(toolStart).Milliseconds()
			if err != nil {
				collector.finishError(idx, startedAt, dur, err.Error())
				// Return error as JSON string but DO NOT fail the agent loop.
				return string(mustJSON(map[string]any{"error": err.Error()})), nil
			}

			collector.finishOK(idx, startedAt, dur, []byte(outStr))
			return outStr, nil
		}
		out = append(out, t)
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func execNestfulPython(ctx context.Context, cfg ToolRuntimeConfig, toolName string, argsJSON []byte, outputKeys []string) (string, error) {
	// Validate executable dir early.
	stat, err := os.Stat(cfg.ExecutableFuncDir)
	if err != nil {
		return "", fmt.Errorf("executable_func_dir not accessible: %w", err)
	}
	if !stat.IsDir() {
		return "", fmt.Errorf("executable_func_dir is not a directory")
	}

	// Use func_file_map.json when present; fallback to basic_functions.py.
	mapPath := filepath.Join(cfg.ExecutableFuncDir, "func_file_map.json")
	basicPath := filepath.Join(cfg.ExecutableFuncDir, "basic_functions.py")

	py := strings.Join([]string{
		"import os, sys, json, importlib.util",
		"tool = os.environ.get('NESTFUL_TOOL_NAME','')",
		"exec_dir = os.environ.get('NESTFUL_EXEC_DIR','')",
		"output_keys = json.loads(os.environ.get('NESTFUL_OUTPUT_KEYS_JSON','[]'))",
		"args = json.load(sys.stdin)",
		"func_map_path = os.path.join(exec_dir, 'func_file_map.json')",
		"file_name = None",
		"if os.path.exists(func_map_path):",
		"  with open(func_map_path, 'r', encoding='utf-8') as f:",
		"    m = json.load(f)",
		"  file_name = m.get(tool)",
		"if not file_name:",
		"  file_name = 'basic_functions.py'",
		"file_path = os.path.join(exec_dir, file_name)",
		"spec = importlib.util.spec_from_file_location('nestful_exec_mod', file_path)",
		"mod = importlib.util.module_from_spec(spec)",
		"spec.loader.exec_module(mod)",
		"if not hasattr(mod, tool):",
		"  raise Exception(f'function not found: {tool} in {file_name}')",
		"fn = getattr(mod, tool)",
		"res = None",
		"try:",
		"  if isinstance(args, dict):",
		"    res = fn(**args)",
		"  else:",
		"    res = fn(args)",
		"except TypeError:",
		"  # Fallback: positional by arg_<n> ordering if present.",
		"  if isinstance(args, dict):",
		"    def _arg_i(k):",
		"      if k.startswith('arg_'):",
		"        try: return int(k.split('_',1)[1])",
		"        except: return 10**9",
		"      return 10**9",
		"    keys = sorted(list(args.keys()), key=_arg_i)",
		"    res = fn(*[args[k] for k in keys])",
		"  else:",
		"    res = fn(args)",
		"out = None",
		"if isinstance(res, dict):",
		"  out = res",
		"elif len(output_keys) == 1:",
		"  out = { output_keys[0]: res }",
		"elif isinstance(res, (list, tuple)) and len(output_keys) == len(res):",
		"  out = { k: v for k, v in zip(output_keys, res) }",
		"else:",
		"  out = { 'result': res }",
		"sys.stdout.write(json.dumps(out))",
	}, "\n")

	cmd := exec.CommandContext(ctx, cfg.PythonBin, "-c", py)
	cmd.Env = append(os.Environ(),
		"NESTFUL_TOOL_NAME="+toolName,
		"NESTFUL_EXEC_DIR="+cfg.ExecutableFuncDir,
		"NESTFUL_OUTPUT_KEYS_JSON="+string(mustJSON(outputKeys)),
		"NESTFUL_FUNC_FILE_MAP="+mapPath,
		"NESTFUL_BASIC_FUNCS="+basicPath,
	)
	cmd.Stdin = bytes.NewReader(argsJSON)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", errors.New(msg)
	}

	out := strings.TrimSpace(stdout.String())
	if out == "" {
		return "", errors.New("empty tool output")
	}
	// Validate it is JSON.
	var tmp any
	if json.Unmarshal([]byte(out), &tmp) != nil {
		return "", fmt.Errorf("tool output is not valid json: %s", out)
	}
	return out, nil
}

type traceCollector struct {
	mu     sync.Mutex
	next   int
	events []ToolTraceEvent
}

func (t *traceCollector) start(name string, args []byte) (idx int, startedAtMs int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	idx = t.next
	t.next++
	startedAtMs = time.Now().UnixMilli()
	t.events = append(t.events, ToolTraceEvent{
		Index:       idx,
		Name:        name,
		Args:        append([]byte{}, args...),
		Ok:          false,
		DurationMs:  0,
		StartedAtMs: startedAtMs,
	})
	return idx, startedAtMs
}

func (t *traceCollector) finishOK(idx int, startedAtMs int64, durationMs int64, outputJSON []byte) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.events {
		if t.events[i].Index != idx {
			continue
		}
		t.events[i].Ok = true
		t.events[i].DurationMs = durationMs
		t.events[i].StartedAtMs = startedAtMs
		// Store output as JSON when possible.
		var raw json.RawMessage
		if json.Unmarshal(outputJSON, &raw) == nil {
			t.events[i].Output = raw
		} else {
			t.events[i].OutputRaw = string(outputJSON)
		}
		return
	}
}

func (t *traceCollector) finishError(idx int, startedAtMs int64, durationMs int64, errMsg string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.events {
		if t.events[i].Index != idx {
			continue
		}
		t.events[i].Ok = false
		t.events[i].DurationMs = durationMs
		t.events[i].StartedAtMs = startedAtMs
		t.events[i].Error = errMsg
		return
	}
}

func (t *traceCollector) Events() []ToolTraceEvent {
	t.mu.Lock()
	defer t.mu.Unlock()
	cp := make([]ToolTraceEvent, len(t.events))
	copy(cp, t.events)
	return cp
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func httpErr(w http.ResponseWriter, err error, status int) {
	writeJSON(w, status, map[string]any{"error": err.Error()})
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

// --- NESTFUL LLM proxy (single-shot) ---

// LLMProxyGenerateRequest is intended for NESTFUL's eval.py single-shot generation.
// It mirrors the proxy request used earlier in toolman/ptc_test.go.
type LLMProxyGenerateRequest struct {
	Prompts      []string `json:"prompts"`
	Model        string   `json:"model,omitempty"`       // provider/name
	Temperature  *float64 `json:"temperature,omitempty"` // default 0
	MaxTokens    *int     `json:"max_tokens,omitempty"`  // default 1000
	SystemPrompt string   `json:"system_prompt,omitempty"`

	// Optional metadata for logging/correlation.
	TraceID   string `json:"trace_id,omitempty"`
	SampleID  string `json:"sample_id,omitempty"`
	ModelName string `json:"model_name,omitempty"`
}

type LLMProxyGenerateResponse struct {
	Texts []string `json:"texts"`
	Error string   `json:"error,omitempty"`
}

// NewLLMProxyMux builds a local mux with:
// - GET  /health
// - POST /generate
//
// /generate performs single-shot LLM calls via an upstream Bellman server.
func NewLLMProxyMux(upstreamBellmanURL string, upstreamKeyName string, upstreamToken string, defaultModelFQN string) (*http.ServeMux, error) {
	if strings.TrimSpace(upstreamBellmanURL) == "" {
		return nil, fmt.Errorf("upstreamBellmanURL is required")
	}
	if strings.TrimSpace(upstreamToken) == "" {
		return nil, fmt.Errorf("upstreamToken is required")
	}
	if strings.TrimSpace(upstreamKeyName) == "" {
		upstreamKeyName = "test"
	}
	if strings.TrimSpace(defaultModelFQN) == "" {
		defaultModelFQN = "OpenAI/gpt-4o-mini"
	}

	client := bellman.New(upstreamBellmanURL, bellman.Key{Name: upstreamKeyName, Token: upstreamToken})

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	mux.HandleFunc("/generate", NewGenerateHandler(client, defaultModelFQN))

	return mux, nil
}

func NewGenerateHandler(client *bellman.Bellman, defaultModelFQN string) http.HandlerFunc {
	if strings.TrimSpace(defaultModelFQN) == "" {
		defaultModelFQN = "OpenAI/gpt-4o-mini"
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req LLMProxyGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, LLMProxyGenerateResponse{Error: "invalid json: " + err.Error()})
			return
		}
		if len(req.Prompts) == 0 {
			writeJSON(w, http.StatusBadRequest, LLMProxyGenerateResponse{Error: "prompts is required"})
			return
		}

		modelStr := strings.TrimSpace(req.Model)
		if modelStr == "" {
			modelStr = defaultModelFQN
		}
		model, err := parseModelFQN(modelStr)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, LLMProxyGenerateResponse{Error: "invalid model: " + err.Error()})
			return
		}

		temp := 0.0
		if req.Temperature != nil {
			temp = *req.Temperature
		}
		maxTok := 1000
		if req.MaxTokens != nil {
			maxTok = *req.MaxTokens
		}
		sys := req.SystemPrompt

		texts := make([]string, 0, len(req.Prompts))
		for i, p := range req.Prompts {
			p = strings.TrimSpace(p)
			if p == "" {
				texts = append(texts, "")
				continue
			}

			start := time.Now()
			llm := client.Generator().Model(model).System(sys).Temperature(temp).MaxTokens(maxTok)
			resp, err := llm.Prompt(prompt.AsUser(p))
			if err != nil {
				writeJSON(w, http.StatusBadGateway, LLMProxyGenerateResponse{Error: fmt.Sprintf("upstream error (idx=%d): %v", i, err)})
				return
			}
			text, err := resp.AsText()
			if err != nil {
				writeJSON(w, http.StatusBadGateway, LLMProxyGenerateResponse{Error: fmt.Sprintf("upstream non-text (idx=%d): %v", i, err)})
				return
			}
			texts = append(texts, strings.TrimSpace(text))
			log.Printf("/generate ok sample_id=%s trace_id=%s model=%s idx=%d ms=%d", req.SampleID, req.TraceID, model.FQN(), i, time.Since(start).Milliseconds())
		}

		writeJSON(w, http.StatusOK, LLMProxyGenerateResponse{Texts: texts})
	}
}

func parseModelFQN(fqn string) (gen.Model, error) {
	fqn = strings.TrimSpace(fqn)
	provider, name, found := strings.Cut(fqn, "/")
	if !found {
		provider, name, found = strings.Cut(fqn, ".")
	}
	if !found {
		return gen.Model{}, fmt.Errorf("expected provider/name (or provider.name), got %q", fqn)
	}
	provider = canonicalProvider(provider)
	name = canonicalModelName(name)
	if provider == "" || name == "" {
		return gen.Model{}, fmt.Errorf("expected provider/name (or provider.name), got %q", fqn)
	}
	return gen.Model{Provider: provider, Name: name}, nil
}

func canonicalProvider(p string) string {
	pl := strings.ToLower(strings.TrimSpace(p))
	switch pl {
	case "openai":
		return "OpenAI"
	case "vertexai", "vertex":
		return "VertexAI"
	case "anthropic":
		return "Anthropic"
	case "ollama":
		return "Ollama"
	case "vllm":
		return "vLLM"
	case "voyageai", "voyage":
		return "VoyageAI"
	default:
		return strings.TrimSpace(p)
	}
}

func canonicalModelName(n string) string {
	n = strings.TrimSpace(n)
	n = strings.ReplaceAll(n, "_", "-")
	if strings.HasPrefix(n, "gpt4o-") {
		n = "gpt-4o-" + strings.TrimPrefix(n, "gpt4o-")
	}
	return n
}

var errBodyTooLarge = errors.New("request body too large")

func readAllWithLimit(r io.Reader, max int64) ([]byte, error) {
	lr := &io.LimitedReader{R: r, N: max + 1}
	b, err := io.ReadAll(lr)
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > max {
		return nil, errBodyTooLarge
	}
	return b, nil
}
