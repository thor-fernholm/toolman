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
func adaptToolsToJSPTC(vm *goja.Runtime, inputTools []tools.Tool) tools.Tool {
	var descriptions []string

	// Helper to extract keys from schema (naive implementation)
	getArgKeys := func(s *schema.JSON) string {
		if s == nil || len(s.Properties) == 0 {
			return "{}"
		}
		var keys []string
		for k := range s.Properties {
			keys = append(keys, k)
		}
		// Returns format: { question } or { amount, from, to }
		return fmt.Sprintf("{ %s }", strings.Join(keys, ", "))
	}

	// register each tool in the VM and build docs
	for _, t := range inputTools {
		err := bindToolToJSVM(vm, t)
		if err != nil { //TODO: handle error
			fmt.Printf("error occurred: %e\n", err)
			return tools.Tool{}
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

		// execute JS
		res, err := vm.RunString(arg.Code)
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
				"Combine all logic into ONE script and return an object with final results. "+
				docsFragment,
		),
		tools.WithArgSchema(Args{}),
		tools.WithFunction(executor),
	)

	return ptcTool
	//return PTCPackage{
	//	Tool:           ptcTool,
	//	PromptFragment: docsFragment, //TODO: remove or where should docs be used/visible?
	//}
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
