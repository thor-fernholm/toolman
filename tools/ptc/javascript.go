package ptc

import (
	"context"
	"encoding/json"
	"fmt"
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

	// extract param keys from schema (naive implementation) TODO add type?
	getArgKeys := func(s *schema.JSON) string {
		if s == nil || len(s.Properties) == 0 {
			return "{}"
		}
		// collect keys
		var keys []string
		for k := range s.Properties {
			keys = append(keys, k)
		}
		// build "key: type"
		var parts []string
		for _, k := range keys {
			prop := s.Properties[k]
			typeName := "any"
			if prop.Type != "" {
				typeName = fmt.Sprintf("%v", prop.Type)
			}
			parts = append(parts, fmt.Sprintf("%s: %s", k, typeName))
		}
		// Returns format: { question: string } or { amount: number, from: string, to: string }
		return fmt.Sprintf("{ %s }", strings.Join(parts, ", "))
	}

	// register each tool in the VM and build docs
	for _, t := range inputTools {
		err := bindToolToJSVM(vm, t)
		if err != nil {
			return tools.Tool{}, "", fmt.Errorf("error occurred: %w", err)
		}
		// create signature description like: function_name({ argument: type }): description...
		argSig := getArgKeys(t.ArgumentSchema)
		desc := fmt.Sprintf(" - %s(%s): %s", t.Name, argSig, t.Description)
		descriptions = append(descriptions, desc)
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
			"MANDATORY: Execute JavaScript code. Input must be a valid, self-contained JS string.\n"+
				"Important: JS code is a pure function (deterministic). "+
				"Combine all logic into ONE script. "+
				"Returns a JSON object with results.\n"+
				"In JavaScript environment, the following tools are available:\n"+
				docsFragment,
		),
		tools.WithArgSchema(CodeArgs{}),
		tools.WithFunction(executor),
	)

	systemFragment := "\n\n" + getSystemFragmentJS() + "\n## Additional available JS tools (functions):\n" + docsFragment
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

// guardRailJS guardrails code before exec; important since LLMs trained for diff. coding objectives
func guardRailJS(code string) (string, error) { // TODO: add more guardrails
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

You are running inside a minimal, embedded JavaScript runtime (Goja).
This environment is NOT Node.js, NOT a browser, and NOT standard JavaScript.

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

## Mandatory Execution Rules
1. Single IIFE required
All code MUST be wrapped in exactly one Immediately Invoked Function Expression:
(function() { ... })()

2. Synchronous execution only
All tool calls are blocking and synchronous.

3. Single execution
You may call code_execution exactly once per turn.

4. Return-only output
Do not log or print. The returned value is the final output.

## Tool Usage Rules (IMPORTANT)
- File operations must use file-related tools
- Network operations must use network tools
- Data retrieval must use provided fetch tools

Never invent, assume, or simulate APIs.

If no appropriate tool exists for a required operation, return an object
explaining the limitation instead of attempting unsupported JavaScript.

## Correct Example
(function() {
  var file = read_file({ path: "final_report.pdf" });
  write_file({ path: "temp/final_report.pdf", data: file });
  var file_status = get_status({ data: file })
  return { status: file_status };
})()
`
}
