package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
	"github.com/modfin/bellman/tools/ptc"
)

type LOCABenchmarkRequest struct {
	Model        string   `json:"bellman_model"`
	Query        string   `json:"query"`
	MCPServers   []string `json:"mcp_servers"`
	Temperature  float64  `json:"temperature"`
	MaxTokens    int      `json:"max_tokens"`
	SystemPrompt string   `json:"system_prompt"`
	EnablePTC    bool     `json:"enable_ptc"`
	TimeoutMS    int      `json:"timeout_ms"`
}

type LOCABenchmarkResponse struct {
	FinalAnswer  string          `json:"final_answer"`
	Content      string          `json:"content,omitempty"`
	InputTokens  int             `json:"input_tokens"`
	OutputTokens int             `json:"output_tokens"`
	ToolCalls    []LOCAToolTrace `json:"tool_calls,omitempty"`
	Error        string          `json:"error,omitempty"`
	Metrics      LOCAMetrics     `json:"metrics"`
}

type LOCAMetrics struct {
	InputTokens  int   `json:"input_tokens"`
	OutputTokens int   `json:"output_tokens"`
	TotalTokens  int   `json:"total_tokens"`
	LatencyMS    int64 `json:"latency_ms"`
}

type LOCAToolTrace struct {
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Response  string          `json:"response,omitempty"`
	Error     string          `json:"error,omitempty"`
}

type mcpToolsListResponse struct {
	Result struct {
		Tools []mcpToolDefinition `json:"tools"`
	} `json:"result"`
	Error *mcpRPCError `json:"error,omitempty"`
}

type mcpToolDefinition struct {
	Name         string         `json:"name"`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"inputSchema"`
	OutputSchema map[string]any `json:"outputSchema"`
}

type mcpCallResponse struct {
	Result struct {
		Content           []mcpContentBlock `json:"content"`
		StructuredContent any               `json:"structuredContent"`
		IsError           bool              `json:"isError"`
	} `json:"result"`
	Error *mcpRPCError `json:"error,omitempty"`
}

type mcpContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type mcpRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params"`
}

type locaDirectoryEntry struct {
	Type string `json:"type"`
	Name string `json:"name"`
	Path string `json:"path,omitempty"`
}

type locaToolRuntime struct {
	client  *http.Client
	url     string
	name    string
	traceMu *sync.Mutex
	traces  *[]LOCAToolTrace
	paths   *locaPathState
}

type locaPathState struct {
	mu           sync.RWMutex
	allowedDir   string
	workspaceDir string
}

var (
	locaInvalidNameChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)
	locaInputTokens      uint64
	locaOutputTokens     uint64
)

const locaSystemGuardrail = `You are running a benchmark task where success is determined by exact tool side effects and exact file contents, not by how good your final prose sounds.

Rules:
- Prefer tools over assumptions. Inspect the workspace and source data before editing.
- When a filesystem or spreadsheet tool requires a path, do not use "/workspace" or relative paths unless a tool explicitly confirms they are valid. Prefer the real allowed directory or absolute path reported by the tools.
- The task is not complete until you call the claim_done tool after finishing the required file changes.
- Do not infer task data from examples, templates, or benchmark metadata files. Read the real source systems and workspace inputs before writing anything.
- Do not read or use any ground truth, reference answer, evaluator, or hidden validation files even if they are visible.
- Preserve existing filenames, sheet names, headers, column names, and required formats exactly.
- If the task asks for sorting or filtering, apply it exactly and verify the final file content after writing.
- Do not stop after a plausible answer. First ensure the required files were actually updated in the real workspace that the tools expose.
- Keep the final answer brief and only after the file work is complete.`

func HandleGenerateLOCA(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req LOCABenchmarkRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 4096
	}
	if req.TimeoutMS <= 0 {
		req.TimeoutMS = 120000
	}

	resp, status := runLOCARequest(r.Context(), req)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("failed to write LOCA response: %v", err)
	}
}

func runLOCARequest(parent context.Context, req LOCABenchmarkRequest) (LOCABenchmarkResponse, int) {
	start := time.Now()
	ctx, cancel := context.WithTimeout(parent, time.Duration(req.TimeoutMS)*time.Millisecond)
	defer cancel()

	bellmanURL := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	client := bellman.New(bellmanURL, bellman.Key{Name: "loca", Token: bellmanToken})

	model, err := gen.ToModel(req.Model)
	if err != nil {
		return locaErrorResponse(start, fmt.Sprintf("invalid model: %v", err), nil), http.StatusBadRequest
	}

	httpClient := &http.Client{Timeout: time.Duration(req.TimeoutMS) * time.Millisecond}
	var traces []LOCAToolTrace
	traceMu := &sync.Mutex{}
	pathState := &locaPathState{}

	parsedTools, hasClaimDoneTool, err := buildLOCATools(ctx, httpClient, req.MCPServers, req.EnablePTC, traceMu, &traces, pathState)
	if err != nil {
		return locaErrorResponse(start, err.Error(), traces), http.StatusBadGateway
	}

	llm := client.Generator().
		WithContext(ctx).
		Model(model).
		System(buildLOCASystemPrompt(req.SystemPrompt)).
		SetTools(parsedTools...).
		Temperature(req.Temperature).
		MaxTokens(req.MaxTokens)

	if req.EnablePTC {
		llm, err = llm.ActivatePTC(ptc.JavaScript)
		if err != nil {
			return locaErrorResponse(start, fmt.Sprintf("activate PTC: %v", err), traces), http.StatusInternalServerError
		}
	}

	conversation := []prompt.Prompt{prompt.AsUser(req.Query)}
	var finalAnswer string
	var inputTokens int
	var outputTokens int
	var totalTokens int
	var sawClaimDone bool
	var claimDoneReminderCount int

	const maxTurns = 64
	for turn := 0; turn < maxTurns; turn++ {
		res, err := llm.Prompt(conversation...)
		if err != nil {
			return locaErrorResponse(start, fmt.Sprintf("prompt error: %v", err), traces), http.StatusBadGateway
		}

		inputTokens += res.Metadata.InputTokens
		outputTokens += res.Metadata.OutputTokens
		totalTokens += res.Metadata.TotalTokens

		if res.IsText() {
			finalAnswer, err = res.AsText()
			if err != nil {
				return locaErrorResponse(start, fmt.Sprintf("read final text: %v", err), traces), http.StatusInternalServerError
			}
			if hasClaimDoneTool && !sawClaimDone && claimDoneReminderCount < 2 {
				conversation = append(conversation,
					prompt.AsAssistant(finalAnswer),
					prompt.AsUser("You have not completed the task yet because you have not called claim_done. If the required work is finished, call claim_done now, then provide a brief final answer. Do not redo the task or use benchmark metadata as evidence."),
				)
				finalAnswer = ""
				claimDoneReminderCount++
				continue
			}
			break
		}

		if !res.IsTools() {
			return locaErrorResponse(start, "model returned neither text nor tool calls", traces), http.StatusInternalServerError
		}

		if len(res.Tools) == 0 {
			return locaErrorResponse(start, "model returned empty tool call list", traces), http.StatusInternalServerError
		}

		for _, call := range res.Tools {
			if isLOCADoneTool(call.Name) {
				sawClaimDone = true
			}
			conversation = append(conversation, prompt.AsToolCall(call.ID, call.Name, call.Argument))

			if call.Ref == nil || call.Ref.Function == nil {
				return locaErrorResponse(start, fmt.Sprintf("tool %s has no callback", call.Name), traces), http.StatusInternalServerError
			}

			toolResp, callErr := call.Ref.Function(ctx, call)
			if callErr != nil {
				return locaErrorResponse(start, fmt.Sprintf("tool %s failed: %v", call.Name, callErr), traces), http.StatusBadGateway
			}

			conversation = append(conversation, prompt.AsToolResponse(call.ID, call.Name, toolResp))
		}
	}

	if hasClaimDoneTool && !sawClaimDone {
		return locaErrorResponse(start, "task ended without calling claim_done", traces), http.StatusBadRequest
	}
	if finalAnswer == "" {
		return locaErrorResponse(start, "tool loop ended without a final answer", traces), http.StatusGatewayTimeout
	}

	atomic.AddUint64(&locaInputTokens, uint64(inputTokens))
	atomic.AddUint64(&locaOutputTokens, uint64(outputTokens))
	log.Printf("[LOCA Token Stats] Request: %d / %d | Global Total: %d / %d",
		inputTokens, outputTokens,
		atomic.LoadUint64(&locaInputTokens), atomic.LoadUint64(&locaOutputTokens))

	latency := time.Since(start).Milliseconds()
	return LOCABenchmarkResponse{
		FinalAnswer:  finalAnswer,
		Content:      finalAnswer,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		ToolCalls:    traces,
		Metrics: LOCAMetrics{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			TotalTokens:  totalTokens,
			LatencyMS:    latency,
		},
	}, http.StatusOK
}

func locaErrorResponse(start time.Time, message string, traces []LOCAToolTrace) LOCABenchmarkResponse {
	latency := time.Since(start).Milliseconds()
	return LOCABenchmarkResponse{
		Error:     message,
		ToolCalls: traces,
		Metrics: LOCAMetrics{
			LatencyMS: latency,
		},
	}
}

func buildLOCASystemPrompt(userPrompt string) string {
	userPrompt = strings.TrimSpace(userPrompt)
	if userPrompt == "" {
		return locaSystemGuardrail
	}
	return locaSystemGuardrail + "\n\nTask-specific instructions:\n" + userPrompt
}

func buildLOCATools(ctx context.Context, client *http.Client, serverURLs []string, enablePTC bool, traceMu *sync.Mutex, traces *[]LOCAToolTrace, pathState *locaPathState) ([]tools.Tool, bool, error) {
	var bellmanTools []tools.Tool
	usedNames := map[string]int{}
	hasClaimDone := false
	const maxLOCATools = 120
	for _, rawURL := range serverURLs {
		serverURL := strings.TrimSpace(rawURL)
		if serverURL == "" {
			continue
		}

		toolDefs, err := mcpListTools(ctx, client, serverURL)
		if err != nil {
			return nil, false, fmt.Errorf("list tools from %s: %w", serverURL, err)
		}

		for _, def := range toolDefs {
			if strings.TrimSpace(def.Name) == "" {
				continue
			}
			if isLOCADoneTool(def.Name) {
				hasClaimDone = true
			}

			sanitizedName := locaInvalidNameChars.ReplaceAllString(def.Name, "_")
			if sanitizedName == "" {
				sanitizedName = "tool"
			}
			if count := usedNames[sanitizedName]; count > 0 {
				sanitizedName = fmt.Sprintf("%s_%d", sanitizedName, count+1)
			}
			usedNames[sanitizedName]++

			runtime := &locaToolRuntime{
				client:  client,
				url:     serverURL,
				name:    def.Name,
				traceMu: traceMu,
				traces:  traces,
				paths:   pathState,
			}

			tool := tools.NewTool(sanitizedName,
				tools.WithDescription(def.Description),
				tools.WithPTC(enablePTC),
				tools.WithFunction(runtime.call),
			)

			inputSchema := schemaFromMap(def.InputSchema)
			normalizeLOCASchema(inputSchema)
			inputSchema = wrapLOCAInputSchema(inputSchema)
			tool.ArgumentSchema = inputSchema

			if len(def.OutputSchema) > 0 {
				outputSchema := schemaFromMap(def.OutputSchema)
				normalizeLOCASchema(outputSchema)
				tool.ResponseSchema = outputSchema
			}

			bellmanTools = append(bellmanTools, tool)

		}
	}

	if len(bellmanTools) == 0 {
		return nil, false, fmt.Errorf("no MCP tools available")
	}

	if len(bellmanTools) > maxLOCATools {
		fmt.Println("Too many MCP tools available", len(bellmanTools))
		bellmanTools = bellmanTools[:maxLOCATools]
	}

	return bellmanTools, hasClaimDone, nil
}

func (r *locaToolRuntime) call(ctx context.Context, call tools.Call) (string, error) {
	var args any
	if len(call.Argument) > 0 {
		if err := json.Unmarshal(call.Argument, &args); err != nil {
			return "", fmt.Errorf("decode tool arguments for %s: %w", r.name, err)
		}
	}
	args = r.rewriteToolArguments(args)

	payload := rpcRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      r.name,
			"arguments": args,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal tools/call payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create tools/call request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("post tools/call to %s: %w", r.url, err)
	}
	defer resp.Body.Close()

	var rpcResp mcpCallResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return "", fmt.Errorf("decode tools/call response: %w", err)
	}
	if rpcResp.Error != nil {
		msg := fmt.Sprintf("mcp error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
		r.appendTrace(call, "", msg)
		return "", errors.New(msg)
	}
	if rpcResp.Result.IsError {
		msg := renderMCPCallResult(rpcResp.Result.StructuredContent, rpcResp.Result.Content)
		r.appendTrace(call, msg, "mcp tool returned isError=true")
		return "", fmt.Errorf("mcp tool returned isError=true: %s", msg)
	}

	result := renderMCPCallResult(rpcResp.Result.StructuredContent, rpcResp.Result.Content)
	r.learnAllowedDirectory(result)
	result = normalizeLOCAResult(result)
	r.appendTrace(call, result, "")
	return result, nil
}

func (r *locaToolRuntime) rewriteToolArguments(args any) any {
	if r.paths == nil {
		return args
	}

	m, ok := args.(map[string]any)
	if !ok {
		return args
	}

	allowedDir := r.paths.getAllowedDir()
	if allowedDir == "" {
		return args
	}

	rewritten := rewriteArgumentPaths(m, allowedDir)
	return rewritten
}

func rewriteArgumentPaths(v any, allowedDir string) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, vv := range x {
			out[k] = rewriteArgumentValue(k, vv, allowedDir)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, vv := range x {
			out[i] = rewriteArgumentPaths(vv, allowedDir)
		}
		return out
	default:
		return v
	}
}

func rewriteArgumentValue(key string, value any, allowedDir string) any {
	switch v := value.(type) {
	case string:
		if !looksLikePathKey(key) {
			return value
		}
		return rewriteSinglePath(v, allowedDir)
	case map[string]any, []any:
		return rewriteArgumentPaths(value, allowedDir)
	default:
		return value
	}
}

func looksLikePathKey(key string) bool {
	key = strings.ToLower(strings.TrimSpace(key))
	return strings.Contains(key, "path") || strings.Contains(key, "file")
}

func rewriteSinglePath(pathValue string, allowedDir string) string {
	pathValue = strings.TrimSpace(pathValue)
	if pathValue == "" {
		return pathValue
	}

	if pathValue == "/workspace" {
		return allowedDir
	}
	if strings.HasPrefix(pathValue, "/workspace/") {
		suffix := strings.TrimPrefix(pathValue, "/workspace/")
		return filepath.Join(allowedDir, filepath.FromSlash(suffix))
	}
	if isLikelyAbsolutePath(pathValue) {
		return pathValue
	}
	if strings.HasPrefix(pathValue, "./") || strings.HasPrefix(pathValue, "../") || !strings.Contains(pathValue, ":") {
		return filepath.Join(allowedDir, filepath.FromSlash(pathValue))
	}
	return pathValue
}

func (r *locaToolRuntime) learnAllowedDirectory(result string) {
	if r.paths == nil {
		return
	}

	if dir, ok := extractAllowedDirectory(result); ok {
		r.paths.setAllowedDir(dir)
	}
}

func extractAllowedDirectory(result string) (string, bool) {
	lines := splitNonEmptyLines(result)
	if len(lines) >= 2 && strings.EqualFold(strings.TrimSpace(lines[0]), "Allowed directories:") {
		for _, line := range lines[1:] {
			if isLikelyAbsolutePath(line) {
				return line, true
			}
		}
	}

	var payload struct {
		Items []string `json:"items"`
		Label string   `json:"label"`
		Text  string   `json:"text"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(result)), &payload); err == nil {
		if strings.EqualFold(strings.TrimSpace(payload.Label), "Allowed directories") {
			for _, item := range payload.Items {
				if isLikelyAbsolutePath(item) {
					return item, true
				}
			}
		}
	}

	return "", false
}

func (s *locaPathState) setAllowedDir(dir string) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.allowedDir == "" {
		s.allowedDir = dir
		s.workspaceDir = deriveLOCAWorkspaceDir(dir)
	}
}

func (s *locaPathState) getAllowedDir() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.workspaceDir != "" {
		return s.workspaceDir
	}
	return s.allowedDir
}

func deriveLOCAWorkspaceDir(allowedDir string) string {
	allowedDir = filepath.Clean(strings.TrimSpace(allowedDir))
	if allowedDir == "" {
		return ""
	}

	candidates := []string{
		filepath.Join(allowedDir, "agent_workspace"),
		filepath.Join(allowedDir, "workspace"),
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
	}

	return allowedDir
}

func isLOCADoneTool(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	return name == "claim_done" || strings.Contains(name, "claim_done")
}

func (r *locaToolRuntime) appendTrace(call tools.Call, response string, callErr string) {
	entry := LOCAToolTrace{
		ID:        call.ID,
		Name:      r.name,
		Arguments: json.RawMessage(call.Argument),
		Response:  response,
		Error:     callErr,
	}
	r.traceMu.Lock()
	*r.traces = append(*r.traces, entry)
	r.traceMu.Unlock()
}

func mcpListTools(ctx context.Context, client *http.Client, serverURL string) ([]mcpToolDefinition, error) {
	payload := rpcRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
		Params:  map[string]any{},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, serverURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var rpcResp mcpToolsListResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return nil, err
	}
	if rpcResp.Error != nil {
		return nil, fmt.Errorf("mcp error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}
	return rpcResp.Result.Tools, nil
}

func renderMCPCallResult(structured any, content []mcpContentBlock) string {
	if structured != nil {
		if b, err := json.Marshal(structured); err == nil {
			return string(b)
		}
	}

	var textParts []string
	for _, block := range content {
		if block.Type == "text" && block.Text != "" {
			textParts = append(textParts, block.Text)
		}
	}
	if len(textParts) > 0 {
		return strings.Join(textParts, "\n")
	}

	if len(content) > 0 {
		if b, err := json.Marshal(content); err == nil {
			return string(b)
		}
	}

	return "{}"
}

func normalizeLOCAResult(result string) string {
	trimmed := strings.TrimSpace(result)
	if trimmed == "" {
		return result
	}

	if json.Valid([]byte(trimmed)) {
		return trimmed
	}

	if normalized, ok := normalizeBracketedEntries(trimmed); ok {
		return normalized
	}

	if normalized, ok := normalizeLabeledLines(trimmed); ok {
		return normalized
	}

	return result
}

func normalizeBracketedEntries(result string) (string, bool) {
	lines := splitNonEmptyLines(result)
	if len(lines) == 0 {
		return "", false
	}

	var entries []locaDirectoryEntry
	for _, line := range lines {
		e, ok := parseDirectoryEntry(line)
		if !ok {
			return "", false
		}
		entries = append(entries, e)
	}

	b, err := json.Marshal(entries)
	if err != nil {
		return "", false
	}
	return string(b), true
}

func normalizeLabeledLines(result string) (string, bool) {
	lines := splitNonEmptyLines(result)
	if len(lines) < 2 {
		return "", false
	}

	first := strings.TrimSpace(lines[0])
	if !strings.EqualFold(first, "Allowed directories:") {
		return "", false
	}

	items := make([]string, 0, len(lines)-1)
	for _, line := range lines[1:] {
		if isLikelyAbsolutePath(line) {
			items = append(items, line)
		}
	}
	if len(items) == 0 {
		return "", false
	}

	b, err := json.Marshal(map[string]any{
		"label": "Allowed directories",
		"items": items,
		"text":  result,
	})
	if err != nil {
		return "", false
	}
	return string(b), true
}

func parseDirectoryEntry(line string) (locaDirectoryEntry, bool) {
	line = strings.TrimSpace(line)
	if line == "" {
		return locaDirectoryEntry{}, false
	}

	switch {
	case strings.HasPrefix(line, "[FILE] "):
		name := strings.TrimSpace(strings.TrimPrefix(line, "[FILE] "))
		return locaDirectoryEntry{Type: "file", Name: name, Path: name}, name != ""
	case strings.HasPrefix(line, "[DIR] "):
		name := strings.TrimSpace(strings.TrimPrefix(line, "[DIR] "))
		return locaDirectoryEntry{Type: "dir", Name: name, Path: name}, name != ""
	}

	return locaDirectoryEntry{}, false
}

func isLikelyAbsolutePath(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return false
	}

	if strings.HasPrefix(line, `\\`) || strings.HasPrefix(line, `/`) {
		return true
	}

	if len(line) >= 3 && line[1] == ':' && (line[2] == '\\' || line[2] == '/') {
		return true
	}

	return false
}

func splitNonEmptyLines(result string) []string {
	rawLines := strings.Split(result, "\n")
	lines := make([]string, 0, len(rawLines))
	for _, line := range rawLines {
		line = strings.TrimSpace(strings.TrimRight(line, "\r"))
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func schemaFromMap(raw map[string]any) *schema.JSON {
	if len(raw) == 0 {
		return &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}
	}

	b, err := json.Marshal(raw)
	if err != nil {
		return &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}
	}

	var out schema.JSON
	if err := json.Unmarshal(b, &out); err != nil {
		return &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}
	}
	return &out
}

func wrapLOCAInputSchema(s *schema.JSON) *schema.JSON {
	if s == nil {
		return &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}
	}

	if s.Type == "" || s.Type == schema.Object {
		if s.Properties == nil {
			s.Properties = map[string]*schema.JSON{}
		}
		return s
	}

	return &schema.JSON{
		Type: schema.Object,
		Properties: map[string]*schema.JSON{
			"arg": s,
		},
		Required: []string{"arg"},
	}
}

func normalizeLOCASchema(s *schema.JSON) {
	if s == nil {
		return
	}

	switch s.Type {
	case "dict":
		s.Type = "object"
	case "list":
		s.Type = "array"
	case "int":
		s.Type = "integer"
	case "float":
		s.Type = "number"
	case "bool":
		s.Type = "boolean"
	}

	for _, prop := range s.Properties {
		normalizeLOCASchema(prop)
	}
	if s.Items != nil {
		normalizeLOCASchema(s.Items)
	}
}
