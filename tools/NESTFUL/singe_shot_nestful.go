package nestful

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/dop251/goja"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
	"github.com/modfin/bellman/tools/ptc"
)

// --- NESTFUL benchmark adapter (single-shot, with/without PTC) ---

type NestfulBenchmarkRequest struct {
	Model              string  `json:"bellman_model"` // provider/name
	Query              string  `json:"query"`
	Tools              []any   `json:"tools"`
	Temperature        float64 `json:"temperature"`
	MaxTokens          int     `json:"max_tokens"`
	SystemPrompt       string  `json:"system_prompt"`
	EnablePTC          bool    `json:"enable_ptc"`
	ToolChoice         string  `json:"tool_choice,omitempty"`           // auto|required|none
	JSExtractTimeoutMs int     `json:"js_extract_timeout_ms,omitempty"` // default 500
}

type NestfulBenchmarkResponse struct {
	GeneratedText string `json:"generated_text"` // JSON list string, NESTFUL scorer input
	Content       string `json:"content,omitempty"`
	InputTokens   int    `json:"input_tokens"`
	OutputTokens  int    `json:"output_tokens"`
	TotalTokens   int    `json:"total_tokens"`
}

type nestfulToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Regex to find invalid tool-name characters.
var invalidNameChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// NewNestfulHandler exposes a single-shot endpoint that returns predicted tool-call sequences in NESTFUL's format.
//
// - If EnablePTC=false: reads res.Tools and returns [{name,arguments,label}, ...]
// - If EnablePTC=true: expects code_execution; runs JS in goja with interceptors and returns the extracted sequence.
//
// Tools are never executed.
func NewNestfulHandler(client *bellman.Bellman, defaultModelFQN string) http.HandlerFunc {
	if strings.TrimSpace(defaultModelFQN) == "" {
		defaultModelFQN = "OpenAI/gpt-4o-mini"
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req NestfulBenchmarkRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpErr(w, fmt.Errorf("invalid json: %w", err), http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Query) == "" {
			httpErr(w, fmt.Errorf("query is required"), http.StatusBadRequest)
			return
		}
		if req.MaxTokens <= 0 {
			req.MaxTokens = 1000
		}
		if req.JSExtractTimeoutMs <= 0 {
			req.JSExtractTimeoutMs = 5000
		}
		choice := strings.ToLower(strings.TrimSpace(req.ToolChoice))
		if choice == "" {
			choice = "required"
		}

		modelStr := strings.TrimSpace(req.Model)
		if modelStr == "" {
			modelStr = defaultModelFQN
		}
		model, err := parseModelFQN(modelStr)
		if err != nil {
			httpErr(w, fmt.Errorf("invalid model: %w", err), http.StatusBadRequest)
			return
		}

		parsedTools, nameMap, err := parseNestfulTools(req.Tools)
		if err != nil {
			httpErr(w, fmt.Errorf("invalid tools: %w", err), http.StatusBadRequest)
			return
		}
		for i := range parsedTools {
			parsedTools[i].UsePTC = req.EnablePTC
			// Never executed; just to keep tool refs non-nil.
			parsedTools[i].Function = func(ctx context.Context, call tools.Call) (string, error) { return "{}", nil }
		}

		llm := client.Generator().
			Model(model).
			System(req.SystemPrompt).
			SetTools(parsedTools...).
			SetPTCLanguage(tools.JavaScript).
			Temperature(req.Temperature).
			MaxTokens(req.MaxTokens)

		switch choice {
		case "required":
			llm = llm.SetToolConfig(tools.RequiredTool)
		case "auto":
			llm = llm.SetToolConfig(tools.AutoTool)
		case "none":
			llm = llm.SetToolConfig(tools.NoTool)
		default:
			httpErr(w, fmt.Errorf("invalid tool_choice: %q", req.ToolChoice), http.StatusBadRequest)
			return
		}

		res, err := llm.Prompt(prompt.AsUser(req.Query))
		println("LMM resp", res, err)
		if err != nil {
			httpErr(w, fmt.Errorf("upstream error: %w", err), http.StatusBadGateway)
			return
		}

		generated, content := nestfulGeneratedText(res, parsedTools, nameMap, req.JSExtractTimeoutMs)
		writeJSON(w, http.StatusOK, NestfulBenchmarkResponse{
			GeneratedText: generated,
			Content:       content,
			InputTokens:   res.Metadata.InputTokens,
			OutputTokens:  res.Metadata.OutputTokens,
			TotalTokens:   res.Metadata.TotalTokens,
		})
	}
}

func parseNestfulTools(raw []any) ([]tools.Tool, map[string]string, error) {
	// nameMap: sanitized -> original
	nameMap := map[string]string{}
	parsed := make([]tools.Tool, 0, len(raw))
	for _, rt := range raw {
		b, err := json.Marshal(rt)
		if err != nil {
			return nil, nil, err
		}
		var def nestfulToolDef
		if err := json.Unmarshal(b, &def); err != nil {
			return nil, nil, err
		}
		if strings.TrimSpace(def.Name) == "" {
			continue
		}
		orig := def.Name
		sanitized := invalidNameChars.ReplaceAllString(orig, "_")
		nameMap[sanitized] = orig

		s := &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}
		var required []string
		for k, v := range def.Parameters {
			ps := schemaFromAny(v)
			if ps == nil {
				ps = &schema.JSON{}
			}
			s.Properties[k] = ps
			if isRequired(v) {
				required = append(required, k)
			}
		}
		if len(required) > 0 {
			sort.Strings(required)
			s.Required = required
		}

		parsed = append(parsed, tools.Tool{
			Name:           sanitized,
			Description:    def.Description,
			ArgumentSchema: s,
		})
	}
	return parsed, nameMap, nil
}

func nestfulGeneratedText(res *gen.Response, availableTools []tools.Tool, nameMap map[string]string, timeoutMs int) (generated string, content string) {
	if !res.IsTools() {
		text, _ := res.AsText()
		return "[]", text
	}
	out := make([]map[string]any, 0)
	errMsgs := make([]string, 0, 1)
	for _, tc := range res.Tools {
		if tc.Name == "code_execution" {
			var codeArgs struct {
				Code string `json:"code"`
			}
			if err := json.Unmarshal(tc.Argument, &codeArgs); err != nil {
				errMsgs = append(errMsgs, fmt.Sprintf("code_execution args unmarshal error: %v", err))
				continue
			}
			seq, errMsg := executeAndExtractNestful(codeArgs.Code, availableTools, timeoutMs)
			if errMsg != "" {
				errMsgs = append(errMsgs, errMsg)
			}
			out = append(out, seq...)
			continue
		}
		args := map[string]any{}
		_ = json.Unmarshal(tc.Argument, &args)
		name := tc.Name
		if orig, ok := nameMap[name]; ok {
			name = orig
		}
		out = append(out, map[string]any{"name": name, "arguments": args})
	}
	for i := range out {
		out[i]["label"] = fmt.Sprintf("$var_%d", i+1)
	}
	return string(mustJSON(out)), strings.Join(errMsgs, "\n")
}

func executeAndExtractNestful(jsCode string, availableTools []tools.Tool, timeoutMs int) ([]map[string]any, string) {
	vm := goja.New()
	var captured []map[string]any

	// Dummy value that survives property access and calls.
	_, _ = vm.RunString(`var __dummy = function(){ return __dummy; }; __dummy = new Proxy(__dummy, { get: function(){ return __dummy; }, apply: function(){ return __dummy; } });`)
	dummyVal := vm.Get("__dummy")

	guarded, guardErr := ptc.GuardRailJS(jsCode)
	if guardErr != nil {
		return captured, fmt.Sprintf("code_execution guardrail error: %v", guardErr)
	}

	timer := time.AfterFunc(time.Duration(timeoutMs)*time.Millisecond, func() {
		vm.Interrupt("timeout")
	})
	defer timer.Stop()

	for _, t := range availableTools {
		tName := t.Name
		interceptor := func(call goja.FunctionCall) goja.Value {
			argsMap := make(map[string]any)
			if len(call.Arguments) > 0 {
				first := call.Arguments[0].Export()
				if obj, ok := first.(map[string]any); ok {
					for k, v := range obj {
						argsMap[k] = v
					}
				} else {
					argsMap["__arg_0__"] = first
					for i := 1; i < len(call.Arguments); i++ {
						argsMap[fmt.Sprintf("__arg_%d__", i)] = call.Arguments[i].Export()
					}
				}
			}
			captured = append(captured, map[string]any{"name": tName, "arguments": argsMap})
			return dummyVal
		}
		_ = vm.Set(tName, interceptor)
	}

	if _, err := vm.RunString(guarded); err != nil {
		return captured, fmt.Sprintf("code_execution run error: %v", err)
	}
	fmt.Println(guarded)
	fmt.Println(captured)
	return captured, ""
}

func isRequired(pdef any) bool {
	m, ok := pdef.(map[string]any)
	if !ok {
		return false
	}
	v, ok := m["required"]
	if !ok {
		return false
	}
	b, ok := v.(bool)
	return ok && b
}

func schemaFromAny(v any) *schema.JSON {
	m, ok := v.(map[string]any)
	if !ok {
		if s, ok := v.(string); ok {
			js := &schema.JSON{}
			applyTypeFromString(js, s)
			return js
		}
		return nil
	}

	js := &schema.JSON{}
	if d, ok := m["description"].(string); ok {
		js.Description = d
	}
	if n, ok := m["nullable"].(bool); ok {
		js.Nullable = n
	}
	if typ, ok := m["type"]; ok {
		if ts, ok := typ.(string); ok {
			applyTypeFromString(js, ts)
		}
	}
	if js.Type == schema.Array {
		if items, ok := m["items"]; ok {
			js.Items = schemaFromAny(items)
			if js.Items == nil {
				js.Items = &schema.JSON{}
			}
		}
	}
	if js.Type == schema.Object {
		if props, ok := m["properties"].(map[string]any); ok {
			js.Properties = map[string]*schema.JSON{}
			for k, pv := range props {
				ps := schemaFromAny(pv)
				if ps == nil {
					ps = &schema.JSON{}
				}
				js.Properties[k] = ps
			}
		}
		if ap, ok := m["additionalProperties"]; ok {
			js.AdditionalProperties = schemaFromAny(ap)
		}
	}
	return js
}

func applyTypeFromString(js *schema.JSON, t string) {
	ls := strings.ToLower(strings.TrimSpace(t))
	switch {
	case ls == "dict" || ls == "object":
		js.Type = schema.Object
	case ls == "array" || strings.Contains(ls, "list"):
		js.Type = schema.Array
		js.Items = &schema.JSON{}
	case ls == "integer" || ls == "int":
		js.Type = schema.Integer
	case ls == "number" || ls == "float":
		js.Type = schema.Number
	case ls == "boolean" || ls == "bool":
		js.Type = schema.Boolean
	case ls == "string":
		js.Type = schema.String
	default:
		// leave empty (permissive)
	}
}
