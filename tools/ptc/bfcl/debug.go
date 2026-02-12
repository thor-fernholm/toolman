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
	LLMRawContent  []string    `json:"llm_raw_content"`
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

// THIS IS THE ORIGINAL WORKING LOGIC
func extractLLMContent(hist []prompt.Prompt) []string {
	if len(hist) == 0 {
		fmt.Printf("[debug] no hist")
		return []string{""}
	}

	var response []string

	// Scan backwards
	for i := len(hist) - 1; i >= 0; i-- {
		p := hist[i]

		// 1. Assistant Role (Text / Thoughts / PTC Code)
		if p.Role == prompt.AssistantRole {
			text := fmt.Sprintf("Assistant: %s\n", p.Text)
			response = append(response, text)
			break
		}

		// 2. Tool Call Role (Native Tool Calls)
		if p.Role == prompt.ToolCallRole && p.ToolCall != nil {
			// p.ToolCall.Arguments is []byte, cast to string
			text := fmt.Sprintf("Tool Call: %s\nArguments: %s", p.ToolCall.Name, string(p.ToolCall.Arguments))
			response = append(response, text)
		}

		// 3. Stop if we hit previous turn
		if p.Role == prompt.UserRole || p.Role == prompt.ToolResponseRole {
			break
		}
	}

	if len(response) == 0 {
		fmt.Printf("[debug] no llm output found")
		return []string{"No LLM output found"}
	}
	fmt.Printf("[debug] response: %s\n", response)
	return response
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
