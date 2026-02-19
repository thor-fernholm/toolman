package main

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
	Endpoint       string      `json:"endpoint"` // Add this field: "BFCL" or "CFB"
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

func MiddlewareDebugLogger(endpointName string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		bodyBytes, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		rw := &responseWriterInterceptor{ResponseWriter: w, statusCode: 200}

		next(rw, r)

		go func() {
			// Unmarshal Request (Generic map)
			var reqMap map[string]interface{}
			_ = json.Unmarshal(bodyBytes, &reqMap)

			// Prepare Unified Log Data
			var (
				inputTokens, outputTokens int
				extractedTools            interface{}
				rawContent                []string
				responseMap               map[string]interface{} // For full payload view
			)

			// Unmarshal generic response for the "Full Response" UI view
			_ = json.Unmarshal(rw.body.Bytes(), &responseMap)

			// Switch Logic based on Endpoint
			if endpointName == "CFB" {
				// --- HANDLE OPENAI FORMAT (CFB) ---
				type CfbCompletion struct {
					Choices []struct {
						Message struct {
							Content   string      `json:"content"`
							ToolCalls interface{} `json:"tool_calls"`
						} `json:"message"`
					} `json:"choices"`
					Usage struct {
						PromptTokens     int `json:"prompt_tokens"`
						CompletionTokens int `json:"completion_tokens"`
					} `json:"usage"`
				}
				type CfbResponse struct {
					Completion     CfbCompletion   `json:"completion"`
					ToolmanHistory []prompt.Prompt `json:"toolman_history"`
					ToolmanCalls   []prompt.Prompt `json:"toolman_calls"`
				}
				var resp CfbResponse
				_ = json.Unmarshal(rw.body.Bytes(), &resp)

				inputTokens = resp.Completion.Usage.PromptTokens
				outputTokens = resp.Completion.Usage.CompletionTokens
				rawContent = extractLLMContent(resp.ToolmanHistory)

				if len(resp.Completion.Choices) > 0 {
					extractedTools = resp.Completion.Choices[0].Message.ToolCalls
				}

			} else {
				// --- HANDLE BFCL FORMAT (DEFAULT) ---
				type BfclResponse struct {
					ToolCalls    interface{}     `json:"tool_calls"`
					History      []prompt.Prompt `json:"toolman_history"`
					InputTokens  int             `json:"input_tokens"`
					OutputTokens int             `json:"output_tokens"`
				}
				var resp BfclResponse
				_ = json.Unmarshal(rw.body.Bytes(), &resp)

				inputTokens = resp.InputTokens
				outputTokens = resp.OutputTokens
				extractedTools = resp.ToolCalls
				rawContent = extractLLMContent(resp.History)
			}

			// Update Global Stats
			atomic.AddUint64(&Store.GlobalInputTokens, uint64(inputTokens))
			atomic.AddUint64(&Store.GlobalOutputTokens, uint64(outputTokens))

			Store.Lock()
			defer Store.Unlock()

			// Session Management (Heuristic: New session if no history or first msg)
			reqHist, _ := reqMap["toolman_history"].([]interface{})
			msgs, _ := reqMap["messages"].([]interface{})

			// CFB doesn't use toolman_history in request, so we check messages length
			isNewSession := false
			if endpointName == "CFB" {
				// A rough heuristic for CFB: simple user query usually starts a test
				isNewSession = (len(msgs) == 1)
			} else {
				// BFCL heuristic
				isNewSession = (len(reqHist) == 0)
			}

			if Store.CurrentSess == nil || isNewSession {
				newSess := &Session{
					ID:        fmt.Sprintf("[%s] Test #%d", endpointName, len(Store.Sessions)+1),
					StartTime: time.Now().Format("15:04:05"),
					Requests:  make([]*LogEntry, 0),
				}
				Store.Sessions = append(Store.Sessions, newSess)
				Store.CurrentSess = newSess
			}

			// Create Entry
			entry := &LogEntry{
				Endpoint:       endpointName,
				ID:             len(Store.CurrentSess.Requests) + 1,
				Timestamp:      time.Now().Format("15:04:05.000"),
				RequestJSON:    reqMap,
				ResponseJSON:   responseMap,
				UserQuery:      extractUserQuery(msgs), // This works for both as both use "messages"
				LLMRawContent:  rawContent,
				ExtractedTools: extractedTools,
				InputTokens:    inputTokens,
				OutputTokens:   outputTokens,
				Duration:       time.Since(start).String(),
			}
			Store.CurrentSess.Requests = append(Store.CurrentSess.Requests, entry)
		}()
	}
}

func extractLLMContent(hist []prompt.Prompt) []string {
	if len(hist) == 0 {
		fmt.Printf("[debug] no hist")
		return []string{""}
	}

	var response []string

	// Scan backwards
	hitToolCall := false
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
			hitToolCall = true
		}

		// 3. Stop if we hit previous turn
		if p.Role == prompt.UserRole {
			break
		}

		// stop on first BFCL response (before last tool call) <-- keep LLM output on JS errors...
		if p.Role == prompt.ToolResponseRole && hitToolCall {
			break
		}
	}

	if len(response) == 0 {
		//fmt.Printf("[debug] no llm output found")
		return []string{"No LLM output found"}
	}
	//fmt.Printf("[debug] response: %s\n", response)
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
