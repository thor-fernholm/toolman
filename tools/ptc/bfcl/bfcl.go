package bfcl

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
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
		// 1. safe unmarshal
		if err := json.Unmarshal(call.Argument, &args); err != nil {
			return "", err
		}

		// 2. build string manually, avoiding fmt.Sprintf("%v") on complex nested maps
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
func GetToolCalls(res *gen.Response, availableTools []tools.Tool) []ExtractedCall {
	var calls []ExtractedCall

	if !res.IsTools() {
		return calls
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
				execResult, err := ExecuteAndExtract(codeArgs.Code, availableTools)
				if err == nil {
					// Append all calls found in the JS code
					calls = append(calls, execResult.Calls...)
					// Note: execResult.FinalReturn is available here if you need it later
				} else {
					fmt.Printf("Error extracting PTC calls: %v\n", err)
				}
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

	return calls
}

// ExtractedCall: The structure of a single tool call found during execution
type ExtractedCall map[string]map[string]interface{}

// ExecutionResult holds both the calls found and the final return value of the script
type ExecutionResult struct {
	Calls       []ExtractedCall `json:"tool_calls"`
	FinalReturn interface{}     `json:"final_return"`
}

func ExecuteAndExtract(jsCode string, availableTools []tools.Tool) (*ExecutionResult, error) {
	vm := goja.New()
	var capturedCalls []ExtractedCall

	for _, tool := range availableTools {
		tName := tool.Name

		// INTERCEPTOR: Runs when JS calls a tool
		interceptor := func(call goja.FunctionCall) goja.Value {
			argsMap := make(map[string]interface{})
			// Extract Arguments
			if len(call.Arguments) > 0 {
				if obj, ok := call.Arguments[0].Export().(map[string]interface{}); ok {
					for k, v := range obj {
						// CHANGE: Force wrap the value in a list
						//argsMap[k] = ensureList(v)
						argsMap[k] = v
					}
				}
			}

			// Record the call
			capturedCalls = append(capturedCalls, ExtractedCall{
				tName: argsMap,
			})

			// Return a "Happy Path" Mock Object <-- needed?!
			// This ensures 'var user = ...' gets a value, and 'if (user.is_premium)' works.
			// In a real agent, you might want to inject specific mocks per tool.
			mockReturn := map[string]interface{}{
				"id":         "mock_id_123", // satisfies generic usage
				"is_premium": true,          // HIT THE LOGIC BRANCH (Example 2)
				"amount":     5000,          // HIT THE LOGIC BRANCH (Example 1: > 1000)
				"status":     "success",     // generic field
				"result":     "mock_string", // generic field
			}

			return vm.ToValue(mockReturn)
		}

		if err := registerFunctionPath(vm, tName, interceptor); err != nil {
			return nil, fmt.Errorf("failed to register tool %s: %w", tName, err)
		}
	}

	val, err := vm.RunString(jsCode)
	if err != nil {
		return nil, fmt.Errorf("runtime error: %w", err)
	}

	finalRet := val.Export()

	return &ExecutionResult{
		Calls:       capturedCalls,
		FinalReturn: finalRet,
	}, nil
}

// ... registerFunctionPath (Same as before) ...
func registerFunctionPath(vm *goja.Runtime, path string, fn func(goja.FunctionCall) goja.Value) error {
	parts := strings.Split(path, ".")
	if len(parts) == 1 {
		vm.Set(path, fn)
		return nil
	}
	currentObj := vm.GlobalObject()
	for i := 0; i < len(parts)-1; i++ {
		part := parts[i]
		val := currentObj.Get(part)
		if val == nil || goja.IsUndefined(val) {
			newObj := vm.NewObject()
			if err := currentObj.Set(part, newObj); err != nil {
				return err
			}
			currentObj = newObj
		} else {
			obj := val.ToObject(vm)
			if obj == nil {
				return fmt.Errorf("namespace collision: %s is not an object", part)
			}
			currentObj = obj
		}
	}
	return currentObj.Set(parts[len(parts)-1], fn)
}

func FormatAsPythonCall(name string, args map[string]interface{}) string {
	var parts []string
	for k, v := range args {
		valStr := ""

		switch val := v.(type) {
		case string:
			valStr = fmt.Sprintf("'%s'", val)
		case bool:
			if val {
				valStr = "True"
			} else {
				valStr = "False"
			}
		case nil:
			valStr = "None"
		default:
			b, _ := json.Marshal(v)
			valStr = string(b)
		}

		if strings.HasPrefix(valStr, "[") && strings.HasSuffix(valStr, "]") {
			var list []interface{}
			if err := json.Unmarshal([]byte(valStr), &list); err == nil && len(list) == 1 {
				inner := list[0]
				switch innerVal := inner.(type) {
				case string:
					valStr = fmt.Sprintf("'%s'", innerVal)
				case bool:
					if innerVal {
						valStr = "True"
					} else {
						valStr = "False"
					}
				default:
					b, _ := json.Marshal(inner)
					valStr = string(b)
				}
			}
		}

		parts = append(parts, fmt.Sprintf("%s=%s", k, valStr))
	}

	// Sort args for stability (optional but good for debugging)
	sort.Strings(parts)

	return fmt.Sprintf("%s(%s)", name, strings.Join(parts, ", "))
}
