package bfcl

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/dop251/goja"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

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
			tools.WithFunction(createEchoFunction(sanitizedName)),
		)

		tool.ArgumentSchema = &paramSchema
		parsedTools = append(parsedTools, tool)
	}

	return parsedTools
}

// safe implementation avoiding recursion traps
func createEchoFunction(name string) func(context.Context, tools.Call) (string, error) {
	return func(ctx context.Context, call tools.Call) (string, error) {
		var args map[string]interface{}
		// safe unmarshal
		if err := json.Unmarshal(call.Argument, &args); err != nil {
			return "", err
		}

		// build string manually, avoiding fmt.Sprintf("%v") on complex nested maps
		// which can trigger recursive String() methods if you have custom types
		var parts []string
		for k, v := range args {
			valStr := "nil"
			if v != nil {
				// use json stringify for complex values to be safe & compliant with python syntax
				if b, err := json.Marshal(v); err == nil {
					// json.marshal adds quotes to strings automatically "val",
					// but for python/bfcl we often prefer single quotes for style,
					// though standard json is usually accepted.
					// simple heuristic:
					valStr = string(b)
				} else {
					valStr = fmt.Sprintf("%v", v)
				}
			}
			parts = append(parts, fmt.Sprintf("%s=%s", k, valStr))
		}

		return fmt.Sprintf("%s(%s)", name, strings.Join(parts, ", ")), nil
	}
}

// GetToolCalls extracts calls in the Ground Truth format: [{"func": {"arg": val}}]
func GetToolCalls(res *gen.Response, availableTools []tools.Tool) ([]ExtractedCall, error) {
	var calls []ExtractedCall

	if !res.IsTools() {
		return calls, nil
	}

	for _, tool := range res.Tools {
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
			}
			continue
		}

		// --- Standard Tool Call ---
		// Map it to the same structure: {"func_name": {"arg": val}}
		var argsMap map[string]interface{}

		// Try unmarshaling argument. If it fails (rare), we skip args or make empty map
		if err := json.Unmarshal(tool.Argument, &argsMap); err != nil {
			fmt.Printf("Warning: Failed to unmarshal args for %s: %v\n", tool.Name, err)
			argsMap = make(map[string]interface{})
		}

		// Construct the entry
		entry := ExtractedCall{
			tool.Name: argsMap,
		}
		calls = append(calls, entry)
	}

	return calls, nil
}

// ExtractedCall: The structure of a single tool call found during execution
type ExtractedCall map[string]map[string]interface{}

// ExecutionResult holds both the calls found and the final return value of the script
type ExecutionResult struct {
	Calls []ExtractedCall `json:"tool_calls"`
	//FinalReturn interface{}     `json:"final_return"`
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

	// POLYFILLS: Prevent ReferenceErrors for common globals TODO remove?
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

			// Return mock to keep script running
			return vm.ToValue("mock_return")
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
			fmt.Printf("JS Runtime Warning: %v\n", err)
		}
	}

	return &ExecutionResult{
		Calls: capturedCalls,
	}
}
