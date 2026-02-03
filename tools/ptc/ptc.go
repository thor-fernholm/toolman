package ptc

import (
	"github.com/dop251/goja"
	"github.com/modfin/bellman/tools"
)

//// PTCPackage holds the generated PTC tool and the system prompt fragment
//type PTCPackage struct {
//	Tool           tools.Tool
//	PromptFragment string
//}

type ProgLang string

const (
	JavaScript ProgLang = "js"
	Python     ProgLang = "python"
	Go         ProgLang = "go"
)

// AdaptToolsToPTC converts a list of Bellman tools into a single PTC tool with code execution environment
func AdaptToolsToPTC(vm *goja.Runtime, inputTools []tools.Tool, progLang ProgLang) ([]tools.Tool, tools.Tool) {
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

	// return if no PTC tools set TODO handle error?
	if len(ptcTools) < 1 {
		return nil, tools.Tool{}
	}

	switch progLang {
	case JavaScript:
		return regularTools, adaptToolsToJSPTC(vm, ptcTools)
	default:
		return regularTools, adaptToolsToJSPTC(vm, ptcTools)
	}
}
