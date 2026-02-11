package bfcl

import (
	"bytes"
	_ "embed" // Required for go:embed
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/modfin/bellman/prompt"
)

// --- EMBEDDED HTML ---
//
//go:embed debug.html
var DebugHTML string

// --- STORE (Moved here so handlers can access it) ---
var Store = &LogStore{
	Sessions: make([]*Session, 0),
}

type LogStore struct {
	sync.RWMutex
	Sessions    []*Session `json:"sessions"`
	CurrentSess *Session   `json:"-"`

	// MOVED TOKENS HERE
	GlobalInputTokens  uint64 `json:"global_input"`
	GlobalOutputTokens uint64 `json:"global_output"`
}

type Session struct {
	ID        string      `json:"id"`
	StartTime string      `json:"start_time"`
	Requests  []*LogEntry `json:"requests"`
}

type LogEntry struct {
	ID             int         `json:"id"`
	Timestamp      string      `json:"timestamp"`
	RequestJSON    interface{} `json:"request_json"`
	ResponseJSON   interface{} `json:"response_json"`
	UserQuery      string      `json:"user_query"`
	LLMRawContent  string      `json:"llm_raw_content"`
	ExtractedTools interface{} `json:"extracted_tools"`
	InputTokens    int         `json:"input_tokens"`
	OutputTokens   int         `json:"output_tokens"`
	Duration       string      `json:"duration"`
}

// --- HANDLERS ---

func HandleDebugUI(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.New("debug").Parse(DebugHTML)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	tmpl.Execute(w, nil)
}

func HandleDebugData(w http.ResponseWriter, r *http.Request) {
	Store.RLock()
	defer Store.RUnlock()

	// Calculate cost (approximate)
	in := atomic.LoadUint64(&Store.GlobalInputTokens)
	out := atomic.LoadUint64(&Store.GlobalOutputTokens)
	cost := (float64(in)*0.15 + float64(out)*0.60) / 1_000_000 // uses GPT 4o mini pricing!

	data := map[string]interface{}{
		"sessions":      Store.Sessions,
		"global_input":  in,
		"global_output": out,
		"total_cost":    fmt.Sprintf("$%.4f", cost),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func HandleDebugClear(w http.ResponseWriter, r *http.Request) {
	Store.Lock()
	Store.Sessions = make([]*Session, 0)
	Store.CurrentSess = nil
	atomic.StoreUint64(&Store.GlobalInputTokens, 0)
	atomic.StoreUint64(&Store.GlobalOutputTokens, 0)
	Store.Unlock()
	w.WriteHeader(http.StatusOK)
}

// --- MIDDLEWARE (Wraps your Benchmark Handler) ---

func MiddlewareDebugLogger(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		bodyBytes, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		rw := &responseWriterInterceptor{ResponseWriter: w, statusCode: 200}

		next(rw, r)

		go func() {
			// 1. Unmarshal Request
			var reqMap map[string]interface{}
			_ = json.Unmarshal(bodyBytes, &reqMap)

			// 2. Unmarshal Response (Strongly Typed for Tokens/History)
			type PartialResponse struct {
				ToolCalls interface{} `json:"tool_calls"`
				// We don't need History here for the session logic anymore
				History      []prompt.Prompt `json:"toolman_history"` // <--- KEY FIX
				InputTokens  int             `json:"input_tokens"`
				OutputTokens int             `json:"output_tokens"`
			}
			var respStruct PartialResponse
			_ = json.Unmarshal(rw.body.Bytes(), &respStruct)

			// Full Response Map for UI
			var respMap map[string]interface{}
			_ = json.Unmarshal(rw.body.Bytes(), &respMap)

			atomic.AddUint64(&Store.GlobalInputTokens, uint64(respStruct.InputTokens))
			atomic.AddUint64(&Store.GlobalOutputTokens, uint64(respStruct.OutputTokens))

			Store.Lock()
			defer Store.Unlock()

			// --- FIX STARTS HERE ---

			// 1. Check the INCOMING request history (from reqMap), not the outgoing response.
			// BFCL sends empty/nil history at the start of a test case.
			reqHist, _ := reqMap["toolman_history"].([]interface{})

			// 2. Check Messages length (New test usually has just 1 user message)
			msgs, _ := reqMap["messages"].([]interface{})

			// CRITICAL: If incoming history is empty, IT IS A NEW SESSION.
			isNewSession := (len(reqHist) == 0)

			// Fallback heuristic: If history is missing but we have messages,
			// ensure we don't accidentally merge if BFCL behaves weirdly.
			if Store.CurrentSess == nil || isNewSession {
				newSess := &Session{
					ID:        fmt.Sprintf("Test Case #%d", len(Store.Sessions)+1),
					StartTime: time.Now().Format("15:04:05"),
					Requests:  make([]*LogEntry, 0),
				}
				Store.Sessions = append(Store.Sessions, newSess)
				Store.CurrentSess = newSess
			}
			// --- FIX ENDS HERE ---

			// Helper to parse the response history for the UI snippet
			rawContent := extractLLMContent(respStruct.History)

			entry := &LogEntry{
				ID:             len(Store.CurrentSess.Requests) + 1,
				Timestamp:      time.Now().Format("15:04:05.000"),
				RequestJSON:    reqMap,
				ResponseJSON:   respMap,
				UserQuery:      extractUserQuery(msgs),
				LLMRawContent:  rawContent,
				ExtractedTools: respStruct.ToolCalls,
				InputTokens:    respStruct.InputTokens,
				OutputTokens:   respStruct.OutputTokens,
				Duration:       time.Since(start).String(),
			}
			Store.CurrentSess.Requests = append(Store.CurrentSess.Requests, entry)
		}()
	}
}

//
//func MiddlewareDebugLogger(next http.HandlerFunc) http.HandlerFunc {
//	return func(w http.ResponseWriter, r *http.Request) {
//		start := time.Now()
//
//		// 1. Capture Request
//		bodyBytes, _ := io.ReadAll(r.Body)
//		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
//
//		// 2. Wrap Response Writer
//		rw := &responseWriterInterceptor{ResponseWriter: w, statusCode: 200}
//
//		// 3. CALL NEXT
//		next(rw, r)
//
//		// 4. Async Log
//		go func() {
//			// Define struct subsets to avoid circular dependencies with main
//			// or just use generic maps for robustness
//			var reqMap map[string]interface{}
//			_ = json.Unmarshal(bodyBytes, &reqMap)
//
//			var respMap map[string]interface{}
//			_ = json.Unmarshal(rw.body.Bytes(), &respMap)
//
//			// 2. Unmarshal Response STRONGLY TYPED for History
//			// We define a partial struct to pull out the exact prompt.Prompt objects
//			type PartialResponse struct {
//				ToolCalls    interface{}     `json:"tool_calls"`
//				History      []prompt.Prompt `json:"toolman_history"` // <--- KEY FIX
//				InputTokens  int             `json:"input_tokens"`
//				OutputTokens int             `json:"output_tokens"`
//			}
//			var respStruct PartialResponse
//			_ = json.Unmarshal(rw.body.Bytes(), &respStruct)
//
//			// Extract Metrics safely from generic maps
//			inTok := 0
//			outTok := 0
//			if v, ok := respMap["input_tokens"].(float64); ok {
//				inTok = int(v)
//			}
//			if v, ok := respMap["output_tokens"].(float64); ok {
//				outTok = int(v)
//			}
//
//			// Update Globals
//			atomic.AddUint64(&Store.GlobalInputTokens, uint64(inTok))
//			atomic.AddUint64(&Store.GlobalOutputTokens, uint64(outTok))
//
//			Store.Lock()
//			defer Store.Unlock()
//
//			// Session Logic: Heuristic to detect new test case
//			// If the request has very few messages (e.g. 1 user msg), it's likely a start
//			msgs, _ := reqMap["messages"].([]interface{})
//			hist := respStruct.History // Use real structs
//
//			if Store.CurrentSess == nil || (len(hist) == 0 && len(msgs) <= 1) {
//				newSess := &Session{
//					ID:        fmt.Sprintf("Test Case #%d", len(Store.Sessions)+1),
//					StartTime: time.Now().Format("15:04:05"),
//					Requests:  make([]*LogEntry, 0),
//				}
//				// Prepend new session to top of list
//				Store.Sessions = append(Store.Sessions, newSess)
//				Store.CurrentSess = newSess
//			}
//
//			// Extract Raw LLM Content (Need to dig into history)
//			rawContent := getRawLLMContentFromHistory(hist)
//
//			entry := &LogEntry{
//				ID:             len(Store.CurrentSess.Requests) + 1,
//				Timestamp:      time.Now().Format("15:04:05.000"),
//				RequestJSON:    reqMap,
//				ResponseJSON:   respMap,
//				UserQuery:      extractUserQuery(msgs),
//				LLMRawContent:  rawContent,
//				ExtractedTools: respMap["tool_calls"],
//				InputTokens:    inTok,
//				OutputTokens:   outTok,
//				Duration:       time.Since(start).String(),
//			}
//			Store.CurrentSess.Requests = append(Store.CurrentSess.Requests, entry)
//		}()
//	}
//}

// THIS IS THE ORIGINAL WORKING LOGIC
func extractLLMContent(hist []prompt.Prompt) string {
	if len(hist) == 0 {
		return ""
	}

	// Scan backwards
	for i := len(hist) - 1; i >= 0; i-- {
		p := hist[i]

		// 1. Assistant Role (Text / Thoughts / PTC Code)
		if p.Role == prompt.AssistantRole {
			return p.Text
		}

		// 2. Tool Call Role (Native Tool Calls)
		if p.Role == prompt.ToolCallRole && p.ToolCall != nil {
			// p.ToolCall.Arguments is []byte, cast to string
			return fmt.Sprintf("Tool Call: %s\nArguments: %s", p.ToolCall.Name, string(p.ToolCall.Arguments))
		}

		// 3. Stop if we hit previous turn
		if p.Role == prompt.UserRole || p.Role == prompt.ToolResponseRole {
			break
		}
	}

	return "No LLM output found"
}

func extractUserQuery(msgs []interface{}) string {
	if len(msgs) == 0 {
		return "Unknown"
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if m, ok := msgs[i].(map[string]interface{}); ok {
			if role, _ := m["role"].(string); role == "user" {
				if content, ok := m["content"].(string); ok {
					return content
				}
			}
		}
	}
	return "Unknown"
}

// Helper Interceptor
type responseWriterInterceptor struct {
	http.ResponseWriter
	statusCode int
	body       bytes.Buffer
}

func (w *responseWriterInterceptor) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}
func (w *responseWriterInterceptor) Write(b []byte) (int, error) {
	w.body.Write(b)
	return w.ResponseWriter.Write(b)
}
