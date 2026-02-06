package ptc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

// ParseJsonSchemaTools converts raw json tool definitions into bellman tools
func ParseJsonSchemaTools(rawTools []interface{}, enablePTC bool) []tools.Tool {
	var parsedTools []tools.Tool

	for _, rt := range rawTools {
		// safely extract fields via marshal/unmarshal
		jsonBytes, _ := json.Marshal(rt)
		var tDef struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		}
		if err := json.Unmarshal(jsonBytes, &tDef); err != nil {
			continue
		}

		// parse parameters into bellman schema
		var paramSchema schema.JSON
		if len(tDef.Parameters) > 0 {
			_ = json.Unmarshal(tDef.Parameters, &paramSchema)
		}

		// create tool and manually inject schema
		tool := tools.NewTool(tDef.Name,
			tools.WithDescription(tDef.Description),
			//tools.WithArgSchema(&paramSchema), // <-- seems to cause crash...
			tools.WithPTC(enablePTC),
			tools.WithFunction(createEchoFunction(tDef.Name)),
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
