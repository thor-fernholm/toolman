package bfcl

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	Model          string          `json:"bellman_model"`
	Messages       []Message       `json:"messages"`
	ToolmanHistory []prompt.Prompt `json:"toolman_history"`
	Tools          []interface{}   `json:"tools"`
	Temperature    float64         `json:"temperature"`
	SystemPrompt   string          `json:"system_prompt"`
	EnablePTC      bool            `json:"enable_ptc"`
}

type Message struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	ToolName string `json:"tool_name"`
	ToolID   string `json:"tool_call_id"`
}

type BenchmarkResponse struct {
	ToolCalls      []ExtractedCall `json:"tool_calls"`
	ToolCallIDs    []string        `json:"tool_call_ids"`
	ToolmanHistory []prompt.Prompt `json:"toolman_history"`
	Content        string          `json:"content"`       // Any thought/text
	InputTokens    int             `json:"input_tokens"`  // Added for tracking
	OutputTokens   int             `json:"output_tokens"` // Added for tracking
}

var (
	GlobalInputTokens  uint64
	GlobalOutputTokens uint64
)

func HandleGenerateBFCL(w http.ResponseWriter, r *http.Request) {
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
	client := bellman.New(bellmanUrl, bellman.Key{Name: "bfcl", Token: bellmanToken})

	bfclTools := ParseJsonSchemaTools(req.Tools, req.EnablePTC)

	toolmanHistory := req.ToolmanHistory

	// count toolman user messages
	toolmanUserCount := 0
	for _, p := range toolmanHistory {
		if p.Role == prompt.UserRole {
			toolmanUserCount++
		}
	}
	// add trailing messages from BFCL
	bfclUserCount := 0
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			// only add new user messages from bfcl (not in toolman hist.)
			bfclUserCount++
			if bfclUserCount > toolmanUserCount {
				toolmanHistory = append(toolmanHistory, prompt.AsUser(m.Content))
			}
		}
	}

	// Add tool response after call!
	var rebuiltHistory []prompt.Prompt
	for i, p := range toolmanHistory {
		switch p.Role {
		case prompt.ToolCallRole:
			rebuiltHistory = append(rebuiltHistory, p)
			// find all corresponding tool results and concatenate
			var concatenatedReturns []string
			for j := 0; j < len(req.Messages); j++ {
				if req.Messages[j].Role == "tool_response" && req.Messages[j].ToolID == p.ToolCall.ToolCallID {
					concatenatedReturns = append(concatenatedReturns, fmt.Sprintf("Function '%s' result: %s.", req.Messages[j].ToolName, req.Messages[j].Content))
				}
			}
			// add JS runtime errors to tool response
			if len(toolmanHistory) > i+1 {
				nextPrompt := toolmanHistory[i+1]
				if nextPrompt.Role == prompt.ToolResponseRole && nextPrompt.ToolResponse.ToolCallID == p.ToolCall.ToolCallID {
					concatenatedReturns = append(concatenatedReturns, nextPrompt.ToolResponse.Response)
				}
			}
			rebuiltHistory = append(rebuiltHistory, prompt.AsToolResponse(p.ToolCall.ToolCallID, p.ToolCall.Name, strings.Join(concatenatedReturns, "\n")))
		case prompt.UserRole:
			rebuiltHistory = append(rebuiltHistory, p)
			//case prompt.AssistantRole: // <-- assistant should only come from toolman response, not added here!
			//	rebuiltHistory = append(rebuiltHistory, p)
		}
	}

	model, err := gen.ToModel(req.Model)
	if err != nil {
		log.Fatalf("error: %e", err)
	}
	//model = openai.GenModel_gpt4_1_mini_250414

	// remove bfcl prompt for PTC - misleading!
	if req.EnablePTC { // TODO: this seems dumb, but need to rewrite system prompt otherwise...
		req.SystemPrompt = "WARNING: You are running a benchmark, which means tool function outputs are NOT assigned to variables. " +
			"You must assume that variables can be reset between turns without warning. " +
			"If you receive new information from a tool function call, you MUST set the variable in the top of the script to make sure you are able to use it." +
			"This means you need to disregard any variable statements or assumptions listed below."
	}

	llm := client.Generator().Model(model).
		System(req.SystemPrompt).
		SetTools(bfclTools...).
		SetPTCLanguage(tools.JavaScript).
		Temperature(req.Temperature)

	res, err := llm.Prompt(rebuiltHistory...)
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
	extractedCalls, toolmanCalls, toolCallIDs, err := GetToolCalls(res, bfclTools)

	// add new toolman calls to conversation history
	toolmanHistory = append(rebuiltHistory, toolmanCalls...)

	resp := BenchmarkResponse{
		ToolCalls:      extractedCalls,
		ToolCallIDs:    toolCallIDs,
		ToolmanHistory: toolmanHistory,
		Content:        "Tool calls generated", // TODO <-- is this used in bfcl?
		InputTokens:    res.Metadata.InputTokens,
		OutputTokens:   res.Metadata.OutputTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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
			Response    json.RawMessage `json:"response"`
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

		paramSchema := parseSchemaRawToJSON(tDef.Parameters)
		responseSchema := parseSchemaRawToJSON(tDef.Response)
		normalizeBFCLSchema(&paramSchema, false)
		normalizeBFCLSchema(&responseSchema, true)

		tool := tools.NewTool(sanitizedName,
			tools.WithDescription(tDef.Description),
			tools.WithPTC(enablePTC),
			tools.WithFunction(
				func(context.Context, tools.Call) (string, error) { return "{}", nil },
			),
		)

		tool.ArgumentSchema = &paramSchema
		//tool.ResponseSchema = &responseSchema // Important: cant use since we cant inject real response from BFCL!!!!!!

		parsedTools = append(parsedTools, tool)
	}

	return parsedTools
}

func parseSchemaRawToJSON(Parameters json.RawMessage) schema.JSON {
	// "dict" -> "object"
	var paramSchema schema.JSON

	if len(Parameters) > 0 {
		var check map[string]interface{}
		if err := json.Unmarshal(Parameters, &check); err == nil {

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

	return paramSchema
}

// normalizeBFCLSchema recursively cleans non-standard types from BFCL datasets
func normalizeBFCLSchema(s *schema.JSON, req bool) { // Replace *schema.JSON with your actual struct type if different
	if s == nil {
		return
	}

	// 1. Fix the Pythonic/BFCL type dialects
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

	// if response --> set all fields to required
	if req && s.Type == "object" && len(s.Properties) > 0 && len(s.Required) == 0 {
		for key := range s.Properties {
			s.Required = append(s.Required, key)
		}
	}

	// Recursively traverse and fix nested properties (for objects)
	for _, prop := range s.Properties {
		normalizeBFCLSchema(prop, req)
	}

	// Recursively traverse and fix array items (for lists/arrays)
	if s.Items != nil {
		normalizeBFCLSchema(s.Items, req)
	}
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

// ExtractedCall: The structure of a single tool call found during execution
type ExtractedCall map[string]map[string]interface{}

// ExecutionResult holds both the calls found and the final return value of the script
type ExecutionResult struct {
	Calls []ExtractedCall `json:"tool_calls"`
	Error error           `json:"error"`
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
	timer := time.AfterFunc(5000*time.Millisecond, func() {
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
			err = fmt.Errorf("javascript runtime error: %s", err)
			fmt.Printf("[warning] JS Runtime Error: %v\n", err)
		}
	}

	return &ExecutionResult{
		Calls: capturedCalls,
		Error: err,
	}
}
