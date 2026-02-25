package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dop251/goja"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

// LOCA-bench endpoint: POST /loca

type locaRequest struct {
	BellmanModel string `json:"bellman_model"`
	Query        string `json:"query"`

	// Optional allow-list of tool names. If empty, all discovered MCP tools are available.
	Tools []string `json:"tools,omitempty"`

	Temperature  float64 `json:"temperature"`
	MaxTokens    int     `json:"max_tokens"`
	SystemPrompt string  `json:"system_prompt"`
	EnablePTC    bool    `json:"enable_ptc"`
	ToolChoice   string  `json:"tool_choice,omitempty"`

	MCPServers []string `json:"mcp_servers"`
	TimeoutMS  int      `json:"timeout_ms"`
}

type locaTraceCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type locaMetrics struct {
	LatencyMS    int64 `json:"latency_ms"`
	InputTokens  int   `json:"input_tokens"`
	OutputTokens int   `json:"output_tokens"`
}

type locaResponse struct {
	BFCLTrace []locaTraceCall `json:"bfcl_trace"`
	ToolTrace []locaTraceCall `json:"tool_trace"`
	PTCCode   string          `json:"ptc_code,omitempty"`
	Final     any             `json:"final_answer,omitempty"`
	Error     string          `json:"error"`
	Metrics   locaMetrics     `json:"metrics"`

	// Extra fields used by the existing debug UI middleware.
	ToolCalls      []locaTraceCall `json:"tool_calls,omitempty"`
	ToolmanHistory []prompt.Prompt `json:"toolman_history,omitempty"`
	InputTokens    int             `json:"input_tokens,omitempty"`
	OutputTokens   int             `json:"output_tokens,omitempty"`
}

// --- Minimal MCP client (JSON-RPC 2.0 over HTTP) ---

type mcpClient struct {
	hc *http.Client
}

type mcpToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type mcpRPCReq struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type mcpRPCResp struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Data    any    `json:"data,omitempty"`
	} `json:"error,omitempty"`
}

func (c *mcpClient) postRPC(ctx context.Context, baseURL string, req mcpRPCReq) (mcpRPCResp, error) {
	// Try base URL as-is, then fallback to baseURL + "/mcp".
	try := []string{strings.TrimRight(baseURL, "/")}
	try = append(try, strings.TrimRight(baseURL, "/")+"/mcp")

	body, err := json.Marshal(req)
	if err != nil {
		return mcpRPCResp{}, err
	}

	var lastErr error
	for _, u := range try {
		hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
		if err != nil {
			lastErr = err
			continue
		}
		hreq.Header.Set("Content-Type", "application/json")
		res, err := c.hc.Do(hreq)
		if err != nil {
			lastErr = err
			continue
		}
		b, _ := io.ReadAll(res.Body)
		_ = res.Body.Close()
		if res.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("mcp status %d: %s", res.StatusCode, string(b))
			continue
		}
		var rr mcpRPCResp
		if err := json.Unmarshal(b, &rr); err != nil {
			// Some servers may return non-RPC JSON; wrap it as result.
			return mcpRPCResp{JSONRPC: "2.0", ID: req.ID, Result: b}, nil
		}
		if rr.Error != nil {
			return rr, fmt.Errorf("mcp error %d: %s", rr.Error.Code, rr.Error.Message)
		}
		return rr, nil
	}
	if lastErr == nil {
		lastErr = errors.New("mcp request failed")
	}
	return mcpRPCResp{}, lastErr
}

func (c *mcpClient) ListTools(ctx context.Context, serverURL string) ([]mcpToolDef, error) {
	rr, err := c.postRPC(ctx, serverURL, mcpRPCReq{JSONRPC: "2.0", ID: 1, Method: "tools/list", Params: map[string]any{}})
	if err != nil {
		return nil, err
	}

	// Expected MCP: { result: { tools: [...] } }
	var v struct {
		Tools []mcpToolDef `json:"tools"`
	}
	if err := json.Unmarshal(rr.Result, &v); err == nil && len(v.Tools) > 0 {
		return v.Tools, nil
	}
	// Fallback: result itself is tools array
	var arr []mcpToolDef
	if err := json.Unmarshal(rr.Result, &arr); err == nil && len(arr) > 0 {
		return arr, nil
	}
	// Fallback: whole payload has tools
	var top struct {
		Tools []mcpToolDef `json:"tools"`
	}
	if err := json.Unmarshal(rr.Result, &top); err == nil && len(top.Tools) > 0 {
		return top.Tools, nil
	}
	return []mcpToolDef{}, nil
}

func (c *mcpClient) CallTool(ctx context.Context, serverURL, toolName string, args map[string]any) (any, error) {
	params := map[string]any{"name": toolName, "arguments": args}
	rr, err := c.postRPC(ctx, serverURL, mcpRPCReq{JSONRPC: "2.0", ID: 2, Method: "tools/call", Params: params})
	if err != nil {
		return nil, err
	}

	// Try to decode known MCP shapes.
	var result any
	if err := json.Unmarshal(rr.Result, &result); err == nil {
		return unwrapAndCompactMCPToolResultLOCA(result), nil
	}
	return string(rr.Result), nil
}

func unwrapAndCompactMCPToolResultLOCA(result any) any {
	// MCP tool servers often return wrappers like:
	// {content:[{type:"text", text:"<json>"}], data:..., is_error:false, structured_content:...}
	// Unwrap to a real JSON value (object/array) to reduce token usage and
	// avoid forcing the model to parse JSON encoded as a string.

	// If it's not a wrapper map, we still compact generically.
	inner := unwrapMCPToolWrapperLOCA(result)
	return compactAnyLOCA(inner)
}

func unwrapMCPToolWrapperLOCA(result any) any {
	m, ok := result.(map[string]any)
	if !ok {
		return result
	}

	// Preserve explicit error wrappers.
	if ie, ok := m["is_error"].(bool); ok && ie {
		// Keep a minimal error payload.
		return map[string]any{
			"is_error": true,
			"content":  m["content"],
			"data":     m["data"],
		}
	}

	// Prefer structured_content if present.
	if sc, ok := m["structured_content"]; ok && sc != nil {
		return sc
	}
	// Prefer data if present (some tools put machine-readable output there).
	if d, ok := m["data"]; ok && d != nil {
		return d
	}
	// Some tools use {"content":"..."}.
	if cs, ok := m["content"].(string); ok {
		if v, ok := parseMaybeJSONLOCA(cs); ok {
			return v
		}
		return cs
	}

	// Fallback: parse JSON embedded in content[0].text
	if content, ok := m["content"].([]any); ok && len(content) > 0 {
		if first, ok := content[0].(map[string]any); ok {
			if txt, ok := first["text"].(string); ok {
				if v, ok := parseMaybeJSONLOCA(txt); ok {
					return v
				}
				return txt
			}
		}
	}
	return result
}

func parseMaybeJSONLOCA(s string) (any, bool) {
	ss := strings.TrimSpace(s)
	if ss == "" {
		return nil, false
	}
	// Only attempt if it looks like JSON.
	if !(strings.HasPrefix(ss, "{") || strings.HasPrefix(ss, "[")) {
		return nil, false
	}
	var v any
	if err := json.Unmarshal([]byte(ss), &v); err != nil {
		return nil, false
	}
	return v, true
}

func compactAnyLOCA(v any) any {
	// Generic, size-based compaction. Goal: reduce token usage while keeping
	// results machine-readable across arbitrary MCP servers.
	const (
		maxDepth    = 4
		maxString   = 4000
		maxArrayLen = 50
		maxMapKeys  = 80
	)
	budget := 1500 // max nodes visited
	return compactAnyInnerLOCA(v, 0, &budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
}

func compactAnyInnerLOCA(v any, depth int, budget *int, maxDepth, maxString, maxArrayLen, maxMapKeys int) any {
	if budget == nil {
		return v
	}
	if *budget <= 0 {
		return "<truncated>"
	}
	*budget--

	if v == nil {
		return nil
	}
	if depth >= maxDepth {
		switch vv := v.(type) {
		case string:
			return compactStringLOCA(vv, maxString)
		case bool, float64, float32, int, int64, int32, uint64, uint32:
			return vv
		default:
			b, _ := json.Marshal(v)
			return compactStringLOCA(string(b), maxString)
		}
	}

	switch vv := v.(type) {
	case string:
		if looksLikeDiffLOCA(vv) {
			return summarizeDiffLOCA(vv)
		}
		return compactStringLOCA(vv, maxString)
	case bool, float64, float32, int, int64, int32, uint64, uint32:
		return vv
	case []any:
		if len(vv) > maxArrayLen {
			vv = vv[:maxArrayLen]
		}
		out := make([]any, 0, len(vv))
		for _, it := range vv {
			out = append(out, compactAnyInnerLOCA(it, depth+1, budget, maxDepth, maxString, maxArrayLen, maxMapKeys))
		}
		return out
	case map[string]any:
		keys := make([]string, 0, len(vv))
		for k := range vv {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if len(keys) > maxMapKeys {
			keys = keys[:maxMapKeys]
		}
		out := make(map[string]any, len(keys))
		for _, k := range keys {
			out[k] = compactAnyInnerLOCA(vv[k], depth+1, budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
		}
		return out
	default:
		// Best-effort via JSON roundtrip into map/slice.
		b, err := json.Marshal(v)
		if err == nil {
			var mm map[string]any
			if err := json.Unmarshal(b, &mm); err == nil && mm != nil {
				return compactAnyInnerLOCA(mm, depth, budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
			}
			var aa []any
			if err := json.Unmarshal(b, &aa); err == nil && aa != nil {
				return compactAnyInnerLOCA(aa, depth, budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
			}
			return compactStringLOCA(string(b), maxString)
		}
		return "<unserializable>"
	}
}

func compactStringLOCA(s string, max int) string {
	ss := strings.TrimSpace(s)
	if len(ss) <= max {
		return ss
	}
	head := ss
	if max > 200 {
		head = ss[:max-200]
	}
	tail := ss
	if len(ss) > 200 {
		tail = ss[len(ss)-200:]
	}
	return head + "\n...<truncated>...\n" + tail
}

func looksLikeDiffLOCA(s string) bool {
	ss := strings.TrimSpace(s)
	if ss == "" {
		return false
	}
	if strings.Contains(ss, "```diff") {
		return true
	}
	if strings.Contains(ss, "\n+++ ") && strings.Contains(ss, "\n--- ") && strings.Contains(ss, "\n@@") {
		return true
	}
	if strings.HasPrefix(ss, "Index: ") {
		return true
	}
	return false
}

func summarizeDiffLOCA(s string) any {
	ss := s
	idx := strings.Index(ss, "Index: ")
	target := ""
	if idx >= 0 {
		line := ss[idx:]
		if nl := strings.IndexAny(line, "\r\n"); nl >= 0 {
			line = line[:nl]
		}
		target = strings.TrimSpace(strings.TrimPrefix(line, "Index: "))
	}
	return map[string]any{
		"ok":     true,
		"kind":   "diff",
		"target": target,
	}
}

// --- Tool bootstrap ---

type toolRegistry struct {
	Tools          []tools.Tool
	ToolNameToURL  map[string]string // Bellman tool name -> server URL
	ToolNameToOrig map[string]string // Bellman tool name -> original MCP tool name
}

func mcpSchemaToBellmanSchema(m map[string]any) (*schema.JSON, error) {
	if m == nil {
		return &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}, nil
	}

	// Bellman's schema.JSON is intentionally minimal; many MCP servers return
	// full JSON Schema where fields vary in shape. Pre-normalize the map to
	// avoid JSON unmarshal failures and keep only what Bellman understands.
	//
	// Notably:
	// - "type" may be an array (e.g. ["string","null"]) -> translate to
	//   type:"string" + nullable:true
	// - "additionalProperties" may be boolean true/false -> drop it (best-effort)
	sanitizeMCPJSONSchemaLOCA(m)

	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	var s schema.JSON
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	normalizeSchemaLOCA(&s)
	if s.Type == "" {
		if len(s.Properties) > 0 {
			s.Type = schema.Object
		}
	}
	return &s, nil
}

func sanitizeMCPJSONSchemaLOCA(v any) {
	switch vv := v.(type) {
	case map[string]any:
		// Handle JSON Schema: type can be string or array (nullable via "null").
		if tv, ok := vv["type"]; ok {
			switch t := tv.(type) {
			case []any:
				nullable := false
				chosen := ""
				for _, e := range t {
					es, ok := e.(string)
					if !ok {
						continue
					}
					if es == "null" {
						nullable = true
						continue
					}
					if chosen == "" {
						chosen = es
					}
				}
				if chosen != "" {
					vv["type"] = chosen
				} else {
					delete(vv, "type")
				}
				if nullable {
					if nb, ok := vv["nullable"].(bool); ok {
						vv["nullable"] = nb || true
					} else {
						vv["nullable"] = true
					}
				}
			}
		}

		// Handle JSON Schema: additionalProperties can be bool or schema.
		if ap, ok := vv["additionalProperties"]; ok {
			if _, ok := ap.(bool); ok {
				// Best-effort: Bellman schema expects an object schema, not a boolean.
				delete(vv, "additionalProperties")
			}
		}

		for _, it := range vv {
			sanitizeMCPJSONSchemaLOCA(it)
		}
	case []any:
		for _, it := range vv {
			sanitizeMCPJSONSchemaLOCA(it)
		}
	}
}

func normalizeSchemaLOCA(s *schema.JSON) {
	if s == nil {
		return
	}
	if strings.EqualFold(string(s.Type), "dict") {
		s.Type = schema.Object
	}
	for _, p := range s.Properties {
		normalizeSchemaLOCA(p)
	}
	if s.Items != nil {
		normalizeSchemaLOCA(s.Items)
	}
	if s.AdditionalProperties != nil {
		normalizeSchemaLOCA(s.AdditionalProperties)
	}
	for _, d := range s.Defs {
		normalizeSchemaLOCA(d)
	}
}

func sanitizeToolName(name string) string {
	// Bellman/OpenAI-like function names should avoid special chars.
	s := strings.TrimSpace(name)
	if s == "" {
		return "tool"
	}
	// Replace non [a-zA-Z0-9_] with underscore
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'
		if ok {
			b = append(b, c)
		} else {
			b = append(b, '_')
		}
	}
	out := strings.ToLower(string(b))
	out = strings.Trim(out, "_")
	for strings.Contains(out, "__") {
		out = strings.ReplaceAll(out, "__", "_")
	}
	if out == "" {
		out = "tool"
	}
	if out[0] >= '0' && out[0] <= '9' {
		out = "get_" + out
	}
	return out
}

func buildRegistry(ctx context.Context, mcpServers []string, allowList []string, client *mcpClient) (*toolRegistry, error) {
	allow := map[string]bool{}
	if len(allowList) > 0 {
		for _, n := range allowList {
			allow[strings.TrimSpace(n)] = true
		}
	}

	reg := &toolRegistry{
		Tools:          []tools.Tool{},
		ToolNameToURL:  map[string]string{},
		ToolNameToOrig: map[string]string{},
	}
	seen := map[string]bool{}

	for _, rawURL := range mcpServers {
		u := strings.TrimSpace(rawURL)
		if u == "" {
			continue
		}
		defs, err := client.ListTools(ctx, u)
		if err != nil {
			return nil, fmt.Errorf("list tools from %s: %w", u, err)
		}
		for _, d := range defs {
			origName := strings.TrimSpace(d.Name)
			if origName == "" {
				continue
			}
			if len(allow) > 0 {
				// allow matches original names
				if !allow[origName] && !allow[sanitizeToolName(origName)] {
					continue
				}
			}
			name := sanitizeToolName(origName)
			if seen[name] {
				// Collision: prefer first server. Ignore duplicates.
				continue
			}
			seen[name] = true
			s, err := mcpSchemaToBellmanSchema(d.InputSchema)
			if err != nil {
				return nil, fmt.Errorf("tool %s schema: %w", origName, err)
			}

			t := tools.NewTool(name, tools.WithDescription(d.Description))
			t.ArgumentSchema = s
			reg.Tools = append(reg.Tools, t)
			reg.ToolNameToURL[name] = u
			reg.ToolNameToOrig[name] = origName
		}
	}

	if len(reg.Tools) == 0 {
		return nil, errors.New("no tools discovered from mcp_servers")
	}
	return reg, nil
}

// --- Handler / runners ---

func HandleGenerateLOCA(w http.ResponseWriter, r *http.Request) {
	fmt.Println("HandleGenerateLOCA")
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()
	var req locaRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.BellmanModel) == "" {
		http.Error(w, "bellman_model is required", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if len(req.MCPServers) == 0 {
		http.Error(w, "mcp_servers is required", http.StatusBadRequest)
		return
	}

	// Safety: some upstream providers reject too-large max_tokens.
	// Clamp to a commonly supported completion limit for OpenAI-compatible models.
	if req.MaxTokens > 16384 {
		req.MaxTokens = 16384
	}

	ctx := r.Context()
	if req.TimeoutMS > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(req.TimeoutMS)*time.Millisecond)
		defer cancel()
	}

	bellmanURL := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	if strings.TrimSpace(bellmanURL) == "" || strings.TrimSpace(bellmanToken) == "" {
		http.Error(w, "BELLMAN_URL and BELLMAN_TOKEN must be set", http.StatusInternalServerError)
		return
	}
	bClient := bellman.New(bellmanURL, bellman.Key{Name: "loca", Token: bellmanToken})

	mcp := &mcpClient{hc: &http.Client{Timeout: 60 * time.Second}}
	reg, err := buildRegistry(ctx, req.MCPServers, req.Tools, mcp)
	if err != nil {
		writeLOCAResponse(w, start, nil, nil, "", nil, 0, 0, err)
		return
	}

	// Help the model avoid sandbox path errors by discovering the allowed
	// workspace root and injecting it into the system prompt.
	injectAllowedWorkspaceRootLOCA(ctx, mcp, reg, &req)

	if req.EnablePTC {
		trace, ptcCode, final, hist, inTok, outTok, err := runLOCAPTC(ctx, bClient, mcp, reg, req)
		writeLOCAResponse(w, start, trace, hist, ptcCode, final, inTok, outTok, err)
		return
	}

	trace, final, hist, inTok, outTok, err := runLOCANormal(ctx, bClient, mcp, reg, req)
	writeLOCAResponse(w, start, trace, hist, "", final, inTok, outTok, err)
}

func injectAllowedWorkspaceRootLOCA(ctx context.Context, mcp *mcpClient, reg *toolRegistry, req *locaRequest) {
	if mcp == nil || reg == nil || req == nil {
		return
	}

	serverURL, ok := findMCPServerForOrigToolLOCA(reg, "list_allowed_directories")
	if !ok {
		// Fall back to instruction-only.
		appendWorkspacePathInstructionLOCA(req, "")
		return
	}

	out, err := mcp.CallTool(ctx, serverURL, "list_allowed_directories", map[string]any{})
	if err != nil {
		appendWorkspacePathInstructionLOCA(req, "")
		return
	}

	root := extractFirstWindowsPathLOCA(out)
	appendWorkspacePathInstructionLOCA(req, root)
}

func findMCPServerForOrigToolLOCA(reg *toolRegistry, origToolName string) (string, bool) {
	if reg == nil {
		return "", false
	}
	needle := strings.TrimSpace(origToolName)
	if needle == "" {
		return "", false
	}
	for sanitized, orig := range reg.ToolNameToOrig {
		if orig == needle {
			if u, ok := reg.ToolNameToURL[sanitized]; ok && strings.TrimSpace(u) != "" {
				return u, true
			}
		}
	}
	// Also accept match on sanitized name.
	san := sanitizeToolName(needle)
	if u, ok := reg.ToolNameToURL[san]; ok && strings.TrimSpace(u) != "" {
		return u, true
	}
	return "", false
}

func appendWorkspacePathInstructionLOCA(req *locaRequest, workspaceRoot string) {
	// Keep this short; it is purely to prevent path-related tool failures.
	root := strings.TrimSpace(workspaceRoot)
	instr := "IMPORTANT (workspace/files): You MUST edit the existing CSV files 'assignment_info.csv' and 'quiz_info.csv' in the allowed workspace. Do NOT create new CSV files. Use only absolute file paths inside the allowed workspace directory. Never use '/' or relative paths like 'assignment_info.csv'."
	if root != "" {
		instr += fmt.Sprintf(" Allowed workspace root: %s. When accessing a file, set path to '%s\\\\<filename>'.", root, root)
		instr += fmt.Sprintf(" The two files you must edit are: '%s\\\\assignment_info.csv' and '%s\\\\quiz_info.csv'.", root, root)
	} else {
		instr += " Call list_allowed_directories first, then prefix all file paths with the returned directory."
	}

	// CSV correctness: prevent "prepend new rows" + leaving stale template/example lines behind.
	instr += " CRITICAL (CSV): Before editing, you MUST read the current contents of BOTH CSV files. Use the EXACT existing header line (line 1) as the header, keep it as the FIRST line, and do NOT invent/rename/reorder columns. Remove any example/template rows that were already in the files (e.g. lines containing '(Example)'). When writing, OVERWRITE the entire file contents (replace the full old text with the full new text) so the final file contains ONLY: the header line + the required data rows. Do not prepend/append snippets. Perform one edit per file that replaces the entire content."

	instr += " IMPORTANT (Canvas IDs): Never call course-specific tools with course_id=0. Always call canvas_list_courses first, then iterate over the returned course IDs when calling canvas_list_assignments/canvas_list_quizzes/canvas_list_announcements."

	s := strings.TrimSpace(req.SystemPrompt)
	if s != "" {
		s += "\n\n"
	}
	req.SystemPrompt = s + instr
}

func extractFirstWindowsPathLOCA(out any) string {
	// Best-effort extraction from common MCP wrapper shapes.
	var text string
	switch v := out.(type) {
	case string:
		text = v
	case map[string]any:
		// Prefer data.content if present.
		if d, ok := v["data"].(map[string]any); ok {
			if c, ok := d["content"].(string); ok {
				text = c
			}
		}
		if text == "" {
			if sc, ok := v["structured_content"].(map[string]any); ok {
				if c, ok := sc["content"].(string); ok {
					text = c
				}
			}
		}
		if text == "" {
			if content, ok := v["content"].([]any); ok && len(content) > 0 {
				if first, ok := content[0].(map[string]any); ok {
					if t, ok := first["text"].(string); ok {
						text = t
					}
				}
			}
		}
	default:
		// Last resort: JSON encode to string.
		b, _ := json.Marshal(v)
		text = string(b)
	}

	// Find the first occurrence of a Windows drive path (e.g. C:\...).
	idx := strings.Index(text, ":\\")
	if idx <= 0 {
		return ""
	}
	start := idx - 1
	// Expand backwards to include drive letter.
	for start > 0 {
		ch := text[start-1]
		if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') {
			start--
			break
		}
		break
	}
	end := idx + 3
	for end < len(text) {
		c := text[end]
		if c == '\n' || c == '\r' || c == '"' {
			break
		}
		end++
	}
	return strings.TrimSpace(text[start:end])
}

func writeLOCAResponse(w http.ResponseWriter, start time.Time, trace []locaTraceCall, hist []prompt.Prompt, ptcCode string, final any, inTok, outTok int, err error) {
	resp := locaResponse{
		BFCLTrace:      trace,
		ToolTrace:      trace,
		PTCCode:        ptcCode,
		Final:          final,
		Error:          "",
		Metrics:        locaMetrics{LatencyMS: time.Since(start).Milliseconds(), InputTokens: inTok, OutputTokens: outTok},
		ToolCalls:      trace,
		ToolmanHistory: hist,
		InputTokens:    inTok,
		OutputTokens:   outTok,
	}
	if err != nil {
		resp.Error = err.Error()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func toolChoiceToConfig(choice string, toolMap map[string]tools.Tool) (*tools.Tool, error) {
	c := strings.TrimSpace(choice)
	if c == "" {
		return nil, nil
	}
	switch strings.ToLower(c) {
	case "auto":
		t := tools.AutoTool
		return &t, nil
	case "required":
		t := tools.RequiredTool
		return &t, nil
	default:
		// specific tool
		if t, ok := toolMap[sanitizeToolName(c)]; ok {
			return &t, nil
		}
		if t, ok := toolMap[c]; ok {
			return &t, nil
		}
		return nil, fmt.Errorf("unknown tool_choice %q", choice)
	}
}

func runLOCANormal(ctx context.Context, client *bellman.Bellman, mcp *mcpClient, reg *toolRegistry, req locaRequest) ([]locaTraceCall, any, []prompt.Prompt, int, int, error) {
	model, err := gen.ToModel(req.BellmanModel)
	if err != nil {
		return nil, nil, nil, 0, 0, err
	}

	trace := make([]locaTraceCall, 0)
	var traceMu sync.Mutex

	// Per-request tool result cache to avoid repeating identical, read-only calls.
	// This is intentionally conservative: only cache tools that look non-mutating.
	cache := map[string]string{}
	var cacheMu sync.Mutex

	toolMap := map[string]tools.Tool{}
	boot := make([]tools.Tool, 0, len(reg.Tools))
	for _, t := range reg.Tools {
		name := t.Name
		serverURL := reg.ToolNameToURL[name]
		orig := reg.ToolNameToOrig[name]
		localTool := t
		localTool.UsePTC = false
		localTool.Function = func(ctx context.Context, call tools.Call) (string, error) {
			var args map[string]any
			if err := json.Unmarshal(call.Argument, &args); err != nil {
				args = map[string]any{}
			}
			traceMu.Lock()
			trace = append(trace, locaTraceCall{Name: orig, Arguments: args})
			traceMu.Unlock()

			// Cache hit?
			if isCacheableToolLOCA(orig) {
				key := toolCacheKeyLOCA(serverURL, orig, args)
				cacheMu.Lock()
				cached, ok := cache[key]
				cacheMu.Unlock()
				if ok {
					return cached, nil
				}
			}

			out, err := mcp.CallTool(ctx, serverURL, orig, args)
			if err != nil {
				return "", err
			}
			b, err := json.Marshal(out)
			if err != nil {
				return "", err
			}
			resp := string(b)
			if isCacheableToolLOCA(orig) {
				key := toolCacheKeyLOCA(serverURL, orig, args)
				cacheMu.Lock()
				cache[key] = resp
				cacheMu.Unlock()
			}
			return resp, nil
		}
		boot = append(boot, localTool)
		toolMap[localTool.Name] = localTool
	}

	g := client.Generator().
		Model(model).
		System(req.SystemPrompt).
		SetTools(boot...).
		WithContext(ctx).
		Temperature(req.Temperature)
	if req.MaxTokens > 0 {
		g = g.MaxTokens(req.MaxTokens)
	}
	if tc, err := toolChoiceToConfig(req.ToolChoice, toolMap); err != nil {
		return trace, nil, nil, 0, 0, err
	} else if tc != nil {
		g = g.SetToolConfig(*tc)
	}

	hist := []prompt.Prompt{prompt.AsUser(req.Query)}
	inTok := 0
	outTok := 0

	for step := 0; step < 20; step++ {
		res, err := g.Prompt(hist...)
		if err != nil {
			return trace, nil, hist, inTok, outTok, err
		}
		inTok += res.Metadata.InputTokens
		outTok += res.Metadata.OutputTokens

		if res.IsText() {
			text, _ := res.AsText()
			hist = append(hist, prompt.AsAssistant(text))
			return trace, text, hist, inTok, outTok, nil
		}
		if !res.IsTools() {
			return trace, nil, hist, inTok, outTok, nil
		}
		for _, c := range res.Tools {
			hist = append(hist, prompt.AsToolCall(c.ID, c.Name, c.Argument))
			tool, ok := toolMap[c.Name]
			if !ok || tool.Function == nil {
				return trace, nil, hist, inTok, outTok, fmt.Errorf("tool %q not found", c.Name)
			}
			out, err := tool.Function(ctx, c)
			fmt.Println("executed tool output: ", out)
			if err != nil {
				return trace, nil, hist, inTok, outTok, fmt.Errorf("tool %q failed: %w", c.Name, err)
			}
			hist = append(hist, prompt.AsToolResponse(c.ID, c.Name, out))
			fmt.Println("LLM info: ", hist)
		}
	}
	return trace, nil, hist, inTok, outTok, fmt.Errorf("max steps reached")
}

func isCacheableToolLOCA(origToolName string) bool {
	// Best-effort heuristic: avoid caching tools that might mutate server state or files.
	n := strings.ToLower(strings.TrimSpace(origToolName))
	if n == "" {
		return false
	}
	// Common mutating verbs.
	mut := []string{"write", "create", "update", "delete", "submit", "enroll", "mark_", "login", "logout", "start_", "publish", "post_"}
	for _, m := range mut {
		if strings.Contains(n, m) {
			return false
		}
	}
	return true
}

func toolCacheKeyLOCA(serverURL, toolName string, args map[string]any) string {
	// encoding/json marshals map keys deterministically.
	b, _ := json.Marshal(args)
	return serverURL + "|" + toolName + "|" + string(b)
}

func runLOCAPTC(ctx context.Context, client *bellman.Bellman, mcp *mcpClient, reg *toolRegistry, req locaRequest) ([]locaTraceCall, string, any, []prompt.Prompt, int, int, error) {
	model, err := gen.ToModel(req.BellmanModel)
	if err != nil {
		return nil, "", nil, nil, 0, 0, err
	}

	trace := make([]locaTraceCall, 0)
	var traceMu sync.Mutex
	ptcCode := ""

	// Per-request cache for non-mutating tool calls executed via JS bindings.
	ptcCache := map[string]any{}
	var ptcCacheMu sync.Mutex

	// Build code_execution tool.
	codeTool := tools.NewTool("code_execution",
		tools.WithDescription("Execute a single JavaScript program and return the final value."),
		tools.WithFunction(func(ctx context.Context, call tools.Call) (string, error) {
			var arg struct {
				Code string `json:"code"`
			}
			if err := json.Unmarshal(call.Argument, &arg); err != nil {
				return "", err
			}
			ptcCode = arg.Code

			timeout := time.Duration(req.TimeoutMS) * time.Millisecond
			if timeout <= 0 {
				timeout = 10 * time.Second
			}

			vm := goja.New()
			timedOut := false
			timer := time.AfterFunc(timeout, func() {
				timedOut = true
				vm.Interrupt("timeout")
			})
			defer timer.Stop()

			// Bind each MCP tool as a global JS function.
			for _, t := range reg.Tools {
				name := t.Name
				serverURL := reg.ToolNameToURL[name]
				orig := reg.ToolNameToOrig[name]
				jsName := name
				_ = vm.Set(jsName, func(fc goja.FunctionCall) goja.Value {
					var args map[string]any
					if len(fc.Arguments) > 0 {
						exp := fc.Arguments[0].Export()
						m, ok := exp.(map[string]any)
						if ok {
							args = m
						} else {
							args = map[string]any{}
						}
					} else {
						args = map[string]any{}
					}

					traceMu.Lock()
					trace = append(trace, locaTraceCall{Name: orig, Arguments: args})
					traceMu.Unlock()

					if isCacheableToolLOCA(orig) {
						key := toolCacheKeyLOCA(serverURL, orig, args)
						ptcCacheMu.Lock()
						cached, ok := ptcCache[key]
						ptcCacheMu.Unlock()
						if ok {
							return vm.ToValue(cached)
						}
					}

					out, err := mcp.CallTool(ctx, serverURL, orig, args)
					if err != nil {
						panic(vm.ToValue(err.Error()))
					}
					if isCacheableToolLOCA(orig) {
						key := toolCacheKeyLOCA(serverURL, orig, args)
						ptcCacheMu.Lock()
						ptcCache[key] = out
						ptcCacheMu.Unlock()
					}
					return vm.ToValue(out)
				})
			}

			v, err := vm.RunString(arg.Code)
			if err != nil {
				if timedOut {
					return "", fmt.Errorf("goja timeout after %s", timeout)
				}
				return "", err
			}
			b, err := json.Marshal(v.Export())
			if err != nil {
				return "", err
			}
			return string(b), nil
		}),
	)
	codeTool.ArgumentSchema = &schema.JSON{
		Type: schema.Object,
		Properties: map[string]*schema.JSON{
			"code": {Type: schema.String, Description: "JavaScript to execute"},
		},
		Required: []string{"code"},
	}

	system := strings.TrimSpace(req.SystemPrompt)
	if system != "" {
		system += "\n\n"
	}
	system += locaPTCSystemInstruction(reg)

	g := client.Generator().
		Model(model).
		System(system).
		SetTools(codeTool).
		SetToolConfig(tools.RequiredTool).
		WithContext(ctx).
		Temperature(req.Temperature)
	if req.MaxTokens > 0 {
		g = g.MaxTokens(req.MaxTokens)
	}

	hist := []prompt.Prompt{prompt.AsUser(req.Query)}
	inTok := 0
	outTok := 0

	for step := 0; step < 6; step++ {
		res, err := g.Prompt(hist...)
		if err != nil {
			return trace, ptcCode, nil, hist, inTok, outTok, err
		}
		inTok += res.Metadata.InputTokens
		outTok += res.Metadata.OutputTokens
		if res.IsText() {
			text, _ := res.AsText()
			hist = append(hist, prompt.AsAssistant(text))
			return trace, ptcCode, text, hist, inTok, outTok, nil
		}
		if !res.IsTools() {
			return trace, ptcCode, nil, hist, inTok, outTok, nil
		}
		for _, c := range res.Tools {
			hist = append(hist, prompt.AsToolCall(c.ID, c.Name, c.Argument))
			out, err := codeTool.Function(ctx, c)
			if err != nil {
				// still return partial trace + ptc_code
				hist = append(hist, prompt.AsToolResponse(c.ID, c.Name, err.Error()))
				return trace, ptcCode, nil, hist, inTok, outTok, err
			}
			hist = append(hist, prompt.AsToolResponse(c.ID, c.Name, out))
		}
	}
	return trace, ptcCode, nil, hist, inTok, outTok, fmt.Errorf("max steps reached")
}

func locaPTCSystemInstruction(reg *toolRegistry) string {
	var b strings.Builder
	b.WriteString("You are running a LOCA function-calling benchmark.\n")
	b.WriteString("You MUST call the tool 'code_execution' exactly once per turn when tools are needed.\n")
	b.WriteString("In code_execution, output JavaScript wrapped like: (function(){ ...; return <object>; })()\n")
	b.WriteString("No top-level return. No async/await.\n")
	b.WriteString("Call tools as global functions: tool_name({ ... }) with exactly one object argument.\n")
	b.WriteString("Treat tool outputs as opaque JSON objects; only use fields you see returned.\n")
	b.WriteString("\nAvailable tool names:\n")
	for _, t := range reg.Tools {
		b.WriteString("- ")
		b.WriteString(t.Name)
		b.WriteString("\n")
	}
	return b.String()
}
