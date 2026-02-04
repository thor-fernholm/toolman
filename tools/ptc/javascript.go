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
		return fmt.Sprintf("{ %s }", strings.Join(keys, ", "))
	}

	// register each tool in the VM and build docs
	for _, t := range inputTools {
		err := bindToolToJSVM(vm, t)
		if err != nil {
			return tools.Tool{}, "", fmt.Errorf("error occurred: %w", err)
		}
		// generate signature like: function_name({ argument })
		argSig := getArgKeys(t.ArgumentSchema)

		// create description like: function_name({ argument }): description...
		desc := fmt.Sprintf("- %s(%s): %s", t.Name, argSig, t.Description)
		descriptions = append(descriptions, desc)
	}

	// define the schema for the PTC tool itself
	type Args struct {
		Code string `json:"code" json-description:"Executable JavaScript code using the provided functions."`
	}

	// create the execution function
	executor := func(ctx context.Context, call tools.Call) (string, error) {
		var arg Args
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
			return fmt.Sprintf(`{"error": %q}`, err.Error()), nil
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
			"MANDATORY: Executable JavaScript code ONLY. "+
				"Combine all logic into ONE script and return an object with final results.\n"+
				"In the JS environment the following functions are available:\n"+
				docsFragment,
		),
		tools.WithArgSchema(Args{}),
		tools.WithFunction(executor),
	)

	systemFragment := "\n\n" + getSystemFragmentJS() + "\n\n" + docsFragment
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
	return `# JavaScript Execution Environment
You have access to a **synchronous** JavaScript runtime (goja). Use the 'code_execution' tool to solve the user's request. Assume you have all functions needed to perform the user's task'.

## Strict Execution Rules
1. **MANDATORY SYNTAX (IIFE):** You MUST wrap your entire script in an Immediately Invoked Function Expression: '(function() { ... })()'.
   - *Reason:* This allows you to use variables, loops, and 'return' statements safely.
   - *Warning:* Do NOT start your script with '{'. The runtime interprets this as a block, not an object, causing a syntax error.
2. **SYNCHRONOUS ONLY:** The runtime is blocking. Usage of 'async', 'await', 'Promise' (including 'Promise.all') or '.then()' is strictly FORBIDDEN and will cause a crash.
3. **NO CONSOLE.LOG:** The 'console' object does not exist. Usage of 'console.log' OR 'print'' will crash the runtime. Return data via the function return only.
4. **SINGLE TURN:** Fetch all data and perform all logic in a SINGLE execution.
5. **CALL LIMIT**: You may call code_execution ONLY ONCE per turn, so combine all tasks into a single script. You are penalized for calling code_execution multiple times.

## Correct Usage Example 1
// 1. Wrap logic in (function() { ... })()
(function() {
  // 2. Assign tool results to variables (Direct synchronous calls)
  var joke = askBellman(CONFIG.url, CONFIG.token, "tell me a joke");
  var usdAmount = convert_currency({ amount: 100, from: 'USD', to: 'SEK' });

  // 3. Perform standard JS logic if needed
  var isExpensive = usdAmount > 1000;

  // 4. RETURN the final result object
  return {
    joke_text: joke,
    conversion_result: usdAmount,
    analysis: isExpensive ? "Too expensive" : "Good price"
  };
})()

## Correct Usage Example 2
({
  joke: askBellman("tell me a joke"),
  stock: getStock(id_134508u236)
})
`
}
