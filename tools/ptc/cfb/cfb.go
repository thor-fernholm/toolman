package cfb

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
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"github.com/dop251/goja"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

type BenchmarkRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	//ToolmanHistory []prompt.Prompt `json:"toolman_history"`
	Tools       []interface{} `json:"tools"`
	Temperature float64       `json:"temperature"`
	//SystemPrompt   string          `json:"system_prompt"`
	//ToolChoice string `json:"tool_choice"`
	MaxTokens int  `json:"max_tokens"`
	EnablePTC bool `json:"enable_ptc"`
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
	ToolName  string     `json:"tool_name"`
	ToolID    string     `json:"tool_call_id"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type ResponseMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"` // Nullable in JSON, empty string in Go
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // Vital: This is a stringified JSON blob
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

//type BenchmarkResponse struct {
//	Choice Choice `json:"choice"`
//	//ToolCalls   []ExtractedCall `json:"tool_calls"`
//	//ToolCallIDs []string        `json:"tool_call_ids"`
//	//ToolmanHistory []prompt.Prompt `json:"toolman_history"`
//	//Content      string `json:"content"`       // Any thought/text
//	InputTokens  int `json:"input_tokens"`  // Added for tracking
//	OutputTokens int `json:"output_tokens"` // Added for tracking
//}
//
//type Choice struct {
//	Content   string     `json:"content"`
//	ToolCalls []ToolCall `json:"tool_calls"`
//}
//
//type ToolCall struct {
//	ID       string           `json:"id"`
//	Type     string           `json:"type"`
//	Function ToolCallFunction `json:"function"`
//}
//
//type ToolCallFunction struct {
//	Name      string `json:"name"`
//	Arguments string `json:"arguments"`
//}

// ExtractedCall: The structure of a single tool call found during execution
type ExtractedCall map[string]map[string]interface{}

// ExecutionResult holds both the calls found and the final return value of the script
type ExecutionResult struct {
	Calls []ExtractedCall `json:"tool_calls"`
	Error error           `json:"error"`
}

var (
	GlobalInputTokens  uint64
	GlobalOutputTokens uint64
)

func HandleGenerateCFB(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	//PrintRequest(r) // Debug requests

	var req BenchmarkRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	client := bellman.New(bellmanUrl, bellman.Key{Name: "cfb", Token: bellmanToken})

	bfclTools := ParseJsonSchemaTools(req.Tools, req.EnablePTC)

	// parse CFB messages to toolman history
	var toolmanHistory []prompt.Prompt
	for i, m := range req.Messages {
		switch m.Role {
		case "user":
			toolmanHistory = append(toolmanHistory, prompt.AsUser(m.Content))
		case "assistant":
			// tool calls
			if m.Content == "" && m.ToolCalls != nil {
				for _, t := range m.ToolCalls {
					toolmanHistory = append(toolmanHistory, prompt.AsToolCall(t.ID, t.Function.Name, []byte(t.Function.Arguments)))
					// add corresponding tool responses (can be multiple with PTC)
					var concatenatedResponses []string
					for j := i + 1; j < len(req.Messages); j++ {
						if req.Messages[j].Role == "tool" && req.Messages[j].ToolID == t.ID {
							concatenatedResponses = append(concatenatedResponses, fmt.Sprintf("Function '%s' result: %s.", req.Messages[j].ToolName, req.Messages[j].Content))
						}
					}
					toolmanHistory = append(toolmanHistory, prompt.AsToolResponse(t.ID, t.Function.Name, strings.Join(concatenatedResponses, "\n")))
					// TODO: add JS runtime errors?
				}
			} else { // assistant text response
				toolmanHistory = append(toolmanHistory, prompt.AsAssistant(m.Content))
			}
		case "tool":
			fmt.Println("tool already added...")
			//toolmanHistory = append(toolmanHistory, prompt.AsToolResponse(m.ToolID, m.ToolName, m.Content))
			//toolmanHistory = append(toolmanHistory, prompt.AsToolCall(m.ToolID, m.ToolName, []byte(m.Content)))
		case "observation":
			log.Fatalf("error: unknown type 'observation' in message: %v", m)
			return
			//toolmanHistory = append(toolmanHistory, prompt.AsToolResponse(m.ToolID, m.ToolName, m.Content))
		}
	}

	model, err := gen.ToModel(req.Model)
	if err != nil {
		log.Fatalf("error: %e", err)
	}
	//model = openai.GenModel_gpt4_1_mini_250414

	llm := client.Generator().Model(model).
		System(""). // TODO: check if system prompt available?
		SetTools(bfclTools...).
		SetPTCLanguage(tools.JavaScript).
		Temperature(req.Temperature)

	res, err := llm.Prompt(toolmanHistory...)
	if err != nil {
		log.Printf("Prompt Error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// EXTRACT TOKENS & UPDATE GLOBAL COUNTERS
	inputTokens := res.Metadata.InputTokens
	outputTokens := res.Metadata.OutputTokens

	// Thread-safe increment
	atomic.AddUint64(&GlobalInputTokens, uint64(inputTokens))
	atomic.AddUint64(&GlobalOutputTokens, uint64(outputTokens))

	// Log the running total to the console
	log.Printf("[Token Stats] Request: %d / %d | Global Total: %d / %d",
		inputTokens, outputTokens,
		atomic.LoadUint64(&GlobalInputTokens), atomic.LoadUint64(&GlobalOutputTokens))

	// extract individual new tool calls for bfcl + toolman
	_, toolmanCalls, _, err := GetToolCalls(res, bfclTools)

	// add new toolman calls to conversation history
	//toolmanHistory = append(toolmanHistory, toolmanCalls...)

	var toolCalls []ToolCall
	for _, c := range toolmanCalls {
		tc := ToolCall{
			ID:   c.ToolCall.ToolCallID,
			Type: "function",
			Function: ToolCallFunction{
				Name:      c.ToolCall.Name,
				Arguments: string(c.ToolCall.Arguments),
			},
		}
		toolCalls = append(toolCalls, tc)
	}

	content := ""
	if res.IsText() {
		if content, err = res.AsText(); err != nil {
			log.Printf("error: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	finishReason := "stop"
	if res.IsTools() {
		finishReason = "tool_calls"
	}

	// TODO: add JS runtime errors?
	resp := ChatCompletionResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model.String(),
		Choices: []Choice{{
			Index: 0,
			Message: ResponseMessage{
				Role:      "assistant",
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		},
		},
		Usage: Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	//resp := BenchmarkResponse{
	//	Choice: Choice{
	//		Content:   toolmanCalls[0].Text,
	//		ToolCalls: toolCalls,
	//	},
	//	InputTokens:  res.Metadata.InputTokens,
	//	OutputTokens: res.Metadata.OutputTokens,
	//}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func PrintRequest(r *http.Request) {
	bodyBytes, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	var prettyJSON bytes.Buffer
	if err := json.Indent(&prettyJSON, bodyBytes, "", "  "); err != nil {
		fmt.Printf("Received Raw Body: %s\n", string(bodyBytes))
	} else {
		fmt.Printf("Received Pretty JSON:\n%s\n", prettyJSON.String())
	}
}

// Regex to find invalid characters (only letters, numbers, underscores, dashes allowed)
var invalidNameChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

func ParseJsonSchemaTools(rawTools []interface{}, enablePTC bool) []tools.Tool {
	var parsedTools []tools.Tool

	for _, rt := range rawTools {
		jsonBytes, _ := json.Marshal(rt)

		var tDef struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		}

		// Handle BFCL's nested "function" wrapper if present
		var wrapper struct {
			Function json.RawMessage `json:"function"`
		}
		if err := json.Unmarshal(jsonBytes, &wrapper); err == nil && len(wrapper.Function) > 0 {
			_ = json.Unmarshal(wrapper.Function, &tDef)
		} else {
			_ = json.Unmarshal(jsonBytes, &tDef)
		}

		if tDef.Name == "" {
			continue
		}

		//fmt.Printf("tool (%v) desc: %s", i, tDef.Description)

		// OpenAI rejects dots. "math.factorial" -> "math_factorial"
		sanitizedName := invalidNameChars.ReplaceAllString(tDef.Name, "_")

		// "dict" -> "object"
		var paramSchema schema.JSON

		if len(tDef.Parameters) > 0 {
			var check map[string]interface{}
			if err := json.Unmarshal(tDef.Parameters, &check); err == nil {

				typeVal, _ := check["type"].(string)

				// BFCL uses "dict", OpenAI wants "object"
				if typeVal == "dict" {
					check["type"] = "object"
					typeVal = "object" // Update for the check below
				}

				// If type is NOT object (e.g. "string"), must wrap it
				if typeVal != "" && typeVal != "object" {
					wrapped := map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"arg": check, // Wrap original schema
						},
						"required": []string{"arg"},
					}
					fixedBytes, _ := json.Marshal(wrapped)
					_ = json.Unmarshal(fixedBytes, &paramSchema)
				} else {
					// It's a valid object/dict, but we might have modified "type" in check
					// So we marshal 'check' back, not 'tDef.Parameters'
					fixedBytes, _ := json.Marshal(check)
					_ = json.Unmarshal(fixedBytes, &paramSchema)
				}
			}
		} else {
			// Handle empty parameters
			emptyObj := map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
			b, _ := json.Marshal(emptyObj)
			_ = json.Unmarshal(b, &paramSchema)
		}

		tool := tools.NewTool(sanitizedName,
			tools.WithDescription(tDef.Description),
			tools.WithPTC(enablePTC),
			tools.WithFunction(
				func(context.Context, tools.Call) (string, error) { return "{}", nil },
			),
		)

		tool.ArgumentSchema = &paramSchema
		parsedTools = append(parsedTools, tool)
	}

	return parsedTools
}

// GetToolCalls extracts calls in the Ground Truth format: [{"func": {"arg": val}}]
func GetToolCalls(res *gen.Response, availableTools []tools.Tool) ([]ExtractedCall, []prompt.Prompt, []string, error) {
	// BFCL
	var calls []ExtractedCall
	// Toolman
	var toolCalls []prompt.Prompt
	var toolIDs []string

	if !res.IsTools() { // --> res.IsText()
		text, err := res.AsText()
		if err != nil {
			log.Fatalf("error: %e", err)
		}
		assistant := []prompt.Prompt{prompt.AsAssistant(text)}
		return calls, assistant, toolIDs, nil
	}

	for i, tool := range res.Tools {
		// --- PTC / Code Execution ---
		if tool.Name == "code_execution" {
			var codeArgs struct {
				Code string `json:"code"`
			}
			// Unmarshal the 'argument' string/bytes to get the JS code
			if err := json.Unmarshal(tool.Argument, &codeArgs); err == nil {
				// Run the Extractor
				execResult := ExecuteAndExtract(codeArgs.Code, availableTools)
				// Append all calls found in the JS code
				calls = append(calls, execResult.Calls...)

				// add ID for each bfcl tool call
				for range execResult.Calls {
					toolIDs = append(toolIDs, tool.ID)
				}

				// add toolman call + ID & check for JS execution errors!
				toolCalls = append(toolCalls, prompt.AsToolCall(tool.ID, tool.Name, tool.Argument))
				//toolIDs = append(toolIDs, tool.ID)
				if execResult.Error != nil {
					toolCalls = append(toolCalls, prompt.AsToolResponse(tool.ID, tool.Name, execResult.Error.Error())) // will not be added to bfcl tool calls!
					//toolIDs = append(toolIDs, tool.ID) // <-- don't think this is needed... only for returned bfcl tools
				}
			} else {
				fmt.Printf("Warning: error unmarshalling code_execution argument: %e\n", err)
			}
			continue
		}

		// --- Standard Tool Call ---
		// Map it to the same structure: {"func_name": {"arg": val}}
		var argsMap map[string]interface{}

		// Try unmarshalling argument. If it fails (rare), we skip args or make empty map
		if err := json.Unmarshal(tool.Argument, &argsMap); err != nil {
			fmt.Printf("Warning: Failed to unmarshal args for %s: %v\n", tool.Name, err)
			argsMap = make(map[string]interface{})
		}

		fmt.Printf("Tool call %v: name: %v, args: %v\n", i, tool.Name, tool.Argument)
		toolCalls = append(toolCalls, prompt.AsToolCall(tool.ID, tool.Name, tool.Argument))
		toolIDs = append(toolIDs, tool.ID)

		// Construct the entry
		entry := ExtractedCall{
			tool.Name: argsMap,
		}
		calls = append(calls, entry)
	}

	return calls, toolCalls, toolIDs, nil
}

func ExecuteAndExtract(jsCode string, availableTools []tools.Tool) *ExecutionResult {
	// GLOBAL SAFETY: Recover from any internal Panic
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Critical Panic in Interpreter: %v\n", r)
		}
	}()

	vm := goja.New()
	var capturedCalls []ExtractedCall

	// TIMEOUT SAFETY: Prevent infinite loops (e.g. while(true))
	// Interrupt execution after 500ms.
	timer := time.AfterFunc(500*time.Millisecond, func() {
		vm.Interrupt("timeout")
	})
	defer timer.Stop()

	// POLYFILLS: Prevent ReferenceErrors for common globals
	// LLMs often treat 'console' and 'print' as standard.
	dummyFunc := func(call goja.FunctionCall) goja.Value { return vm.ToValue(nil) }
	vm.Set("print", dummyFunc)

	console := vm.NewObject()
	console.Set("log", dummyFunc)
	console.Set("error", dummyFunc)
	console.Set("warn", dummyFunc)
	vm.Set("console", console)

	for _, tool := range availableTools {
		tName := tool.Name

		// INTERCEPTOR: Runs when JS calls a tool
		interceptor := func(call goja.FunctionCall) goja.Value {
			argsMap := make(map[string]interface{})
			// ROBUST ARG PARSING: Handle any argument style
			if len(call.Arguments) > 0 {
				firstArg := call.Arguments[0].Export()

				// Scenario A: Standard BFCL (First arg is a Dictionary)
				// tool({ "arg": 1 })
				if obj, ok := firstArg.(map[string]interface{}); ok {
					for k, v := range obj {
						argsMap[k] = v
					}
				} else {
					// Scenario B: Hallucinated Positional Args
					// tool("value", 123)
					// We capture them with generic keys so we don't lose data.
					argsMap["__arg_0__"] = firstArg
					for i := 1; i < len(call.Arguments); i++ {
						fmt.Printf("[Fix] caught a previous js extract error...")
						key := fmt.Sprintf("__arg_%d__", i)
						argsMap[key] = call.Arguments[i].Export()
					}
				}
			}

			// Record the call
			capturedCalls = append(capturedCalls, ExtractedCall{
				tName: argsMap,
			})

			// Return generic mock to keep script running
			mock := vm.NewObject()
			mock.Set("status", "success")
			mock.Set("success", true)
			mock.Set("error", nil)

			return mock
		}
		vm.Set(tName, interceptor)
	}

	_, err := vm.RunString(jsCode)

	// GRACEFUL FAILURE
	// If we crash (e.g. syntax error), we STILL return whatever calls we captured.
	if err != nil {
		// Check if it was just our timeout
		var evalErr *goja.InterruptedError
		if !errors.As(err, &evalErr) {
			// If it's a real runtime error, just log it.
			// We DO NOT return the error to the caller, because we want the partial results.
			fmt.Printf("[warning] JS Runtime Error: %v\n", err)
		}
	}

	return &ExecutionResult{
		Calls: capturedCalls,
		Error: fmt.Errorf("javascript runtime error: %s", err),
	}
}
