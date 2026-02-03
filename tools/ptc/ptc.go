package ptc

import (
	"fmt"

	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/tools"
)

// AdaptToolsToPTC converts a list of Bellman tools into a single PTC tool with code execution environment
func AdaptToolsToPTC(r gen.Request, config map[string]string) ([]tools.Tool, []tools.Tool) {
	var regularTools []tools.Tool
	var ptcTools []tools.Tool

	// get PTC enabled tools
	for _, t := range r.Tools {
		if t.UsePTC {
			ptcTools = append(ptcTools, t)
		} else {
			regularTools = append(regularTools, t)
		}
	}

	// return if no PTC tools set TODO handle error?
	if len(ptcTools) < 1 {
		return nil, nil
	}

	switch r.PTCLanguage {
	case gen.JavaScript:
		return regularTools, adaptToolsToJSPTC(ptcTools, config)
	case gen.Python:
		fmt.Println("Python not implemented!")
		return nil, nil
	default:
		return regularTools, adaptToolsToJSPTC(ptcTools, config)
	}
}

// GetSystemFragment returns system prompt fragment for PTC tool "code_execution" TODO put in right place
func GetSystemFragment() string {
	return `\n## JavaScript Environment Functions (Use these ONLY inside code_execution)
You can solve complex logic by writing JavaScript code for the code_execution tool.

## Rules for code_execution
1. CALL LIMIT: You may call code_execution ONLY ONCE per turn. 
2. LOGIC: Perform all calculations and multi-tool logic INSIDE the JS script. Before calling code_execution, plan how to combine all tasks into a single script. You are penalized for making more than one tool call.
3. RETURN: The JS script MUST end with an object containing all final data.
4. FORMAT: Do not use console.log for final data; the last evaluated expression is the return value.
6. SYNTHESIS: Once you have the result from code_execution, you have everything you need. 
7. TERMINATION: Do not call the tool again with the same or modified code.
8. SYNC: Do not use async functions unless specified.

## Example JS Script Input
({
  joke: askBellman(CONFIG.url, CONFIG.token, ""),
  total: Sum(123, 456)
})`
}
