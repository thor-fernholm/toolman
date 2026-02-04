package ptc

import (
	"fmt"

	"github.com/modfin/bellman/tools"
)

// ExtractPTCTools separates regular tools from PTC tools and returns both slices
func ExtractPTCTools(inputTools []tools.Tool) ([]tools.Tool, []tools.Tool) {
	var regularTools []tools.Tool
	var ptcTools []tools.Tool

	// get PTC enabled tools
	for _, t := range inputTools {
		if t.UsePTC {
			ptcTools = append(ptcTools, t)
		} else {
			regularTools = append(regularTools, t)
		}
	}
	return regularTools, ptcTools
}

// AdaptToolsToPTC converts a list of Bellman tools into a single PTC tool with code execution environment
func AdaptToolsToPTC(ptcTools []tools.Tool, language tools.ProgramLanguage) (tools.Tool, string, error) {
	switch language {
	case tools.JavaScript:
		return adaptToolsToJSPTC(ptcTools)
	case tools.Python:
		return tools.Tool{}, "", fmt.Errorf("ptc python not implemented")
	case tools.Go:
		return tools.Tool{}, "", fmt.Errorf("ptc go not implemented")
	default: // default to JS
		return adaptToolsToJSPTC(ptcTools)
	}

}
