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

		code, err := guardRailJS(arg.Code)
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
	docsFragment := strings.Join(descriptions, "\n")

	// create the final PTC tool
	ptcTool := tools.NewTool("code_execution",
		tools.WithDescription(
			"MANDATORY: Execute JavaScript code in a Goja runtime.\n"+
				"Input must be a valid, self-contained JavaScript program wrapped in a single IIFE.\n"+
				"You MUST combine all tool calls and logic into ONE script and call this tool exactly once per turn.\n"+
				"The script MUST return a single value.\n"+
				"Return value must be a valid JSON value (object, array, string, number, boolean, or null).\n"+
				"In the JavaScript environment, the following tool functions are available:\n"+
				docsFragment,
		),
		tools.WithArgSchema(CodeArgs{}),
		tools.WithFunction(executor),
	)

	// create PTC system prompt fragment with tools
	systemFragment := "\n\n" + getSystemFragmentJS() +
		"\n## Return Type Definition\n" +
		"json means any valid JSON value:\n" +
		"- object\n" +
		"- array\n" +
		"- string\n" +
		"- number\n" +
		"- boolean\n" +
		"- null\n\n" +
		"## Available JavaScript Tool Functions (Callable ONLY inside 'code_execution')\n" +
		"Tool signatures below are exact and binding. Do not assume additional parameters, defaults, overloads, or return fields.\n\n" +
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

	exampleArgs := buildExampleArgs(args)

	return fmt.Sprintf(
		`function %s(args: %s): %s

Description:
- %s

Example:
%s(%s)
`,
		t.Name,
		argBlock,
		inferReturnType(t),
		t.Description,
		t.Name,
		exampleArgs,
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

func buildExampleArgs(args []ArgField) string {
	if len(args) == 0 {
		return "{}"
	}

	var parts []string
	for _, a := range args {
		parts = append(parts, fmt.Sprintf("%s: %s", a.Name, exampleValueForType(a.Type)))
	}

	return "{ " + strings.Join(parts, ", ") + " }"
}

func exampleValueForType(t string) string {
	switch t {
	case "string":
		return `"example"`
	case "number":
		return "1"
	case "boolean":
		return "true"
	case "any[]":
		return "[]"
	case "object":
		return "{}"
	default:
		return "null"
	}
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
func guardRailJS(code string) (string, error) { // TODO: add more/update guardrails
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

// getSystemFragmentJS returns system prompt fragment for JS PTC tool "code_execution"
func getSystemFragmentJS() string {
	return `# JavaScript Tool Execution Environment (Goja)

Important:
You must solve the entire user task by writing ONE complete JavaScript program.
All required tool calls, logic, branching, and data processing MUST occur
inside a single JavaScript script executed via code_execution.

You are running inside a minimal, embedded JavaScript runtime (Goja).
This environment is NOT Node.js, NOT a browser, and NOT standard JavaScript.

'code_execution' is a meta-tool used to execute JavaScript.
It is NOT a regular tool function and MUST NOT be emulated or replaced.

Plan all required tool calls first, then write one script that executes them in order.

## Environment Constraints (CRITICAL)
The following APIs and capabilities DO NOT exist:
- require
- import
- fs, path, os, process, Buffer
- DOM, window, or browser APIs
- console or logging
- async functions, promises, or callbacks
- global utilities unless explicitly listed

If a function or object is not explicitly listed in the tool section, it does not exist.
Calling an undefined function will cause execution failure.

## Available Capabilities
You may ONLY:
- Use basic JavaScript syntax (variables, conditionals, loops, arrays, objects)
- Call explicitly listed tool functions
- Return data using return only

All I/O, filesystem access, networking, and side effects MUST be performed
through provided tool functions.

## Mandatory Execution Rules (READ CAREFULLY)
1. ONE SCRIPT
You must write exactly ONE JavaScript program that performs ALL steps of the task.
Do NOT split logic across multiple scripts or turns.

2. ONE code_execution CALL
You must call code_execution exactly once per turn.
All tool calls must be embedded inside that single JavaScript program.

3. SINGLE IIFE
The entire program MUST be wrapped in exactly one Immediately Invoked Function Expression:
(function() { ... })()

4. SYNCHRONOUS ONLY
All tool calls are blocking and synchronous.

5. RETURN-ONLY OUTPUT
Do not log, print, or emit intermediate output.
The single return value is the final result.

Explicitly Forbidden Behavior

- Calling code_execution more than once
- Performing part of the task outside JavaScript
- Calling tools outside the IIFE
- Asking for additional turns to finish the task
- Simulating tool results instead of calling tools

## Tool Usage Rules (IMPORTANT)
- File operations must use file-related tools
- Network operations must use network tools
- Data retrieval must use provided fetch tools
- Always assign tool call results to a variable before using them.

Never invent, assume, or simulate APIs.

If no appropriate tool exists for a required operation, return an object
explaining the limitation instead of attempting unsupported JavaScript.

## Output Contract (CRITICAL)

Your final JavaScript program response MUST be a call to the tool 'code_execution''.

- You MUST NOT output JavaScript code directly as assistant text.
- The JavaScript program MUST be provided as the value of the 'code' argument
  to the 'code_execution' tool.
- The assistant response must consist of exactly ONE tool call:
  code_execution({ code: "<full JavaScript program>" })

Any response that outputs JavaScript outside of a 'code_execution' tool call
is invalid.

## Correct Example (ALL LOGIC IN ONE SCRIPT)
(function() {
  var file = read_file({ path: "final_report.pdf" });
  var status = get_status({ data: file });
  write_file({ path: "temp/final_report.pdf", data: file });
  return { status: status };
})()
`
}
