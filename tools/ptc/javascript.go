package ptc

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/dop251/goja"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

// adaptToolsToJSPTC converts a list of Bellman tools into a single PTC tool with JS execution environment
func adaptToolsToJSPTC(inputTools []tools.Tool) (tools.Tool, string, error) {
	var descriptions []string

	// instantiate goja vm
	vm := goja.New()

	// register each tool in the VM and build docs
	for _, t := range inputTools {
		err := bindToolToJSVM(vm, t)
		if err != nil {
			return tools.Tool{}, "", fmt.Errorf("error occurred: %w", err)
		}
		// create signature description like: function_name({ argument: type }): description...
		signature := formatToolSignature(t)
		descriptions = append(descriptions, signature)
	}

	// define the schema for the PTC tool itself
	type CodeArgs struct {
		Code string `json:"code" json-description:"The executable JavaScript code string."`
	}

	// create the execution function
	executor := func(ctx context.Context, call tools.Call) (string, error) {
		var arg CodeArgs
		if err := json.Unmarshal(call.Argument, &arg); err != nil {
			return "", err
		}

		code, err := GuardRailJS(arg.Code)
		if err != nil {
			return err.Error(), nil
		}

		// execute JS
		res, err := vm.RunString(code)
		if err != nil {
			// return error as JSON so LLM can see it
			return fmt.Sprintf(`{"error": %q. Info: Assigned variables persist in JS Environment...}`, err.Error()), nil
		}

		// export result and marshal
		jsonBytes, err := json.Marshal(res.Export())
		if err != nil {
			return "", err
		}
		return string(jsonBytes), nil
	}

	// tool documentation fragment
	docsFragment := strings.Join(descriptions, "\n\n")

	// create the final PTC tool
	ptcTool := tools.NewTool("code_execution",
		tools.WithDescription(
			"Execute a complete JavaScript program in a minimal Goja runtime.\n"+
				"The input MUST be a self-contained script wrapped exactly as:\n"+
				"(function() { ... })()\n\n"+
				"All task logic and ALL tool calls must be inside this single script.\n"+
				"Call this tool exactly once per turn.\n\n"+
				"The script must return one final JSON value.\n"+
				"No logging, async code, or external APIs are available.\n\n"+
				"The following tool functions are callable inside the script:\n"+
				docsFragment,
		),
		tools.WithArgSchema(CodeArgs{}),
		tools.WithFunction(executor),
	)

	// create PTC system prompt fragment with tools
	systemFragment := "\n\n" + getSystemFragmentJS() +
		"\n## Available JavaScript Tool Functions\n\n" +
		docsFragment

	return ptcTool, systemFragment, nil
}

// bindToolToVM wraps a Bellman tool as a JS function: toolName({ args... })
func bindToolToJSVM(vm *goja.Runtime, t tools.Tool) error {
	wrapper := func(call goja.FunctionCall) goja.Value {
		// check if LLM passed multiple arguments (common mistake)
		if len(call.Arguments) > 1 {
			errMsg := fmt.Sprintf("Error: %s expects a single configuration object argument, but received %d arguments. Usage: %s({ key: val })",
				t.Name, len(call.Arguments), t.Name)
			return vm.ToValue(map[string]string{"error": errMsg})
		}

		// extract JS argument (expecting a single object)
		if len(call.Arguments) == 0 {
			return vm.NewGoError(fmt.Errorf("tool %s requires arguments", t.Name))
		}
		jsArgs := call.Argument(0).Export()

		// marshal args to JSON for the Bellman tool
		jsonArgs, err := json.Marshal(jsArgs)
		if err != nil {
			return vm.NewGoError(err)
		}

		// execute the actual go tool
		// TODO: pass real context if available
		res, err := t.Function(context.Background(), tools.Call{
			Argument: jsonArgs,
		})
		if err != nil {
			// return error string directly so the LLM can self-correct, e.g., "json: cannot unmarshal number..."
			return vm.ToValue(map[string]string{"error": err.Error()})
		}

		// unmarshal result back to JS object if possible
		var parsed interface{}
		if err := json.Unmarshal([]byte(res), &parsed); err == nil {
			return vm.ToValue(parsed)
		}

		// otherwise return raw string
		return vm.ToValue(res)
	}

	err := vm.Set(t.Name, wrapper)
	if err != nil {
		return err
	}

	return nil
}

type ArgField struct {
	Name     string
	Type     string
	Required bool
}

func formatToolSignature(t tools.Tool) string {
	args := extractArgs(t.ArgumentSchema)

	var fields []string
	for _, a := range args {
		name := a.Name
		if !a.Required {
			name += "?"
		}
		fields = append(fields, fmt.Sprintf("  %s: %s", name, a.Type))
	}

	argBlock := ""
	if len(fields) > 0 {
		argBlock = "{\n" + strings.Join(fields, ",\n") + "\n}"
	} else {
		argBlock = "{}"
	}

	//exampleArgs := buildExampleArgs(args)

	return fmt.Sprintf(
		`function %s(args: %s): %s
- %s`,
		t.Name,
		argBlock,
		inferReturnType(t),
		t.Description,
	)
}

func extractArgs(s *schema.JSON) []ArgField {
	if s == nil || len(s.Properties) == 0 {
		return nil
	}

	required := map[string]bool{}
	for _, r := range s.Required {
		required[r] = true
	}

	var args []ArgField
	for name, prop := range s.Properties {
		args = append(args, ArgField{
			Name:     name,
			Type:     mapJSONSchemaType(prop),
			Required: required[name],
		})
	}

	sort.Slice(args, func(i, j int) bool {
		return args[i].Name < args[j].Name
	})

	return args
}

func inferReturnType(t tools.Tool) string {
	// If your Tool type exposes return schema, plug it in here.
	// Otherwise be explicit and conservative.
	return "json"
}

func mapJSONSchemaType(s *schema.JSON) string {
	if s == nil {
		return "unknown"
	}

	switch s.Type {
	case "string":
		return "string"
	case "number", "integer":
		return "number"
	case "boolean":
		return "boolean"
	case "array":
		return "any[]"
	case "object":
		return "object"
	default:
		return "unknown"
	}
}

// guardRailJS guardrails code before exec; important since LLMs trained for diff. coding objectives
func GuardRailJS(code string) (string, error) { // TODO: add more/update guardrails
	if code == "" {
		errMsg := "RuntimeError: No code script provided. Rewrite the code immediately."
		fmt.Printf("[PTC] Blocked empty code attempt\n")
		return code, fmt.Errorf("error: %s", errMsg)
	}

	if strings.Contains(code, "return") && !strings.HasPrefix(strings.TrimSpace(code), "(function") {
		code = fmt.Sprintf("(function() { %s })()", code)
	}

	if strings.Contains(code, "print( ") || strings.Contains(code, "console.log(") {
		errMsg := "RuntimeError: Log functions (e.g., 'console.log' or 'print') are strictly FORBIDDEN in this environment. You must use return data via the function return only. Rewrite the code immediately."
		fmt.Printf("[PTC] Blocked log attempt\n")
		return code, fmt.Errorf("error: %s", errMsg)
	}

	if strings.Contains(code, "async ") || strings.Contains(code, "await") || strings.Contains(code, "async(") {
		errMsg := "RuntimeError: Async functions are strictly FORBIDDEN in this environment. You must use synchronous, blocking calls (e.g., 'const x = tool()', NOT 'await tool()'). Rewrite the code immediately."
		fmt.Printf("[PTC] Blocked async code attempt\n")
		return code, fmt.Errorf("error: %s", errMsg)
	}
	return code, nil
}

func getSystemFragmentJS() string {
	return `# Tool Return Conventions
- Success may return undefined or null.
- Failure returns a string error.
- Absence of a return value means success.
- Always inspect tool return values.

# JavaScript Execution Environment (Goja)

You are generating a single executable program, not a sequence of tool calls.

This environment executes exactly ONE JavaScript program per turn.
You must invoke 'code_execution' at most once.
Inside that program, you may call multiple tool functions as needed.

Execution is part of solving the task.
You may use execution results to compute the final answer.

You run in a minimal embedded JavaScript runtime.
Not Node.js. Not a browser. No async. No console.

Only basic JavaScript syntax is available.
All I/O and side effects must use provided tool functions.
Tool functions are global. Do NOT prefix them with "functions." or any namespace.

## When To Use code_execution

Use 'code_execution' ONLY if the task requires tool functions
(file operations, network, data retrieval, etc).

If no tool is required, respond in natural language.
Never output JavaScript unless invoking 'code_execution'.

## If Using code_execution

Before invoking 'code_execution', determine ALL required tool calls and combine them into ONE complete script.

- Invoke 'code_execution' exactly once per turn.
- Provide one complete script wrapped exactly as:
  (function() { ... })()
- All logic and tool calls must be inside that script.
- Assign tool results to variables before using them.
- Execution is strictly synchronous. Do NOT use async or await.
- You MUST return one final JSON value from the IIFE.
- No logging or external APIs.

If you invoke 'code_execution', your entire response must be
a structured tool call with no text before or after.

Never write code_execution({ ... }) as plain text.
`
}
