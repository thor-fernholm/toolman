package js

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"sort"
	"strings"
	"sync"
	"text/template"
	"time"

	"github.com/dop251/goja"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

type JavaScript struct {
	runtime  *goja.Runtime
	mu       sync.Mutex
	ctx      context.Context // set during Execute, used by tool wrappers
	toolName string
	output   *resultOutput
	Log      *slog.Logger `json:"-"`
}

type resultOutput struct {
	value string
	set   bool
}

type TemplateData struct {
	PTCToolName    string
	Signatures     []FunctionSignatureData
	ReturnFunction string
}

type FunctionSignatureData struct {
	Name          string
	Description   string
	ArgumentNode  *TSNode
	ReturnNode    *TSNode
	UnknownSchema bool
}

// TSNode represents a node in the schema tree, formatted for template rendering.
type TSNode struct {
	Name        string
	Type        string
	Required    bool
	Description string
	Properties  []*TSNode // populated if Type == "object"
	Items       *TSNode   // populated if Type == "array"
	Indent      string
}

//go:embed prompts.tmpl
var templateFS embed.FS
var parsedTemplates *template.Template

const nilValue string = "null"          // nil in JS
const returnFunc string = "__setResult" // define JS return value func

func init() {
	var err error
	parsedTemplates, err = template.ParseFS(templateFS, "prompts.tmpl")
	if err != nil {
		panic(fmt.Errorf("failed to parse prompts.tmpl: %w", err))
	}
}

func NewRuntime(toolName string) (*JavaScript, error) {
	javaScript := &JavaScript{
		runtime:  goja.New(),
		mu:       sync.Mutex{},
		toolName: toolName,
	}
	return javaScript.registerReturn()
}

func (j *JavaScript) Lock() {
	j.mu.Lock()
}

func (j *JavaScript) Unlock() {
	j.mu.Unlock()
}

func (j *JavaScript) Runtime() *goja.Runtime {
	return j.runtime
}

func (j *JavaScript) log(msg string, args ...any) {
	if j.Log == nil {
		return
	}
	j.Log.Debug("[bellman/javascript] "+msg, args...)
}

// AdaptTools converts a list of Bellman tools into a single PTC tool with runtime execution environment
func (j *JavaScript) AdaptTools(tool ...tools.Tool) (tools.Tool, error) {
	for _, t := range tool {
		err := j.bindToolFunction(t)
		if err != nil {
			return tools.Tool{}, fmt.Errorf("error adapting tools to ptc: %w", err)
		}
	}

	type CodeArgs struct {
		Code string `json:"code" json-description:"The executable top-level JavaScript code string."`
	}
	executor := func(ctx context.Context, call tools.Call) (string, error) {
		var arg CodeArgs
		if err := json.Unmarshal(call.Argument, &arg); err != nil {
			return "", err
		}

		res, resErr, err := j.Execute(ctx, arg.Code)
		if err != nil {
			return res, err
		}

		// return error string to LLM
		if resErr != nil {
			return fmt.Sprintf(`{"error": %q}`, resErr.Error()), err
		}

		return res, err
	}

	// create tool description
	var buf bytes.Buffer
	if err := parsedTemplates.ExecuteTemplate(&buf, "ptc_tool_description", TemplateData{ReturnFunction: returnFunc}); err != nil {
		return tools.Tool{}, fmt.Errorf("failed to execute tool description template: %w", err)
	}
	toolDescription := buf.String()

	// create the final PTC tool
	ptcTool := tools.NewTool(j.toolName,
		tools.WithDescription(toolDescription),
		tools.WithArgSchema(CodeArgs{}),
		tools.WithFunction(executor),
	)

	return ptcTool, nil
}

// bindToolFunction wraps a Bellman tool as a runtime function: toolName({ args... })
func (j *JavaScript) bindToolFunction(tool tools.Tool) error {
	escapedName := escapeFunctionName(tool.Name)
	wrapper := func(call goja.FunctionCall) goja.Value {
		// check if LLM passed multiple arguments (common mistake)
		if len(call.Arguments) > 1 {
			errMsg := fmt.Sprintf("Error: %s expects a single configuration object argument, but received %d arguments. Usage: %s({ key: val })",
				escapedName, len(call.Arguments), escapedName)
			return j.runtime.ToValue(map[string]string{"error": errMsg})
		}

		// extract runtime argument (expecting a single object)
		if len(call.Arguments) == 0 {
			return j.runtime.NewGoError(fmt.Errorf("tool %s requires arguments", escapedName))
		}
		jsArgs := call.Argument(0).Export()

		// marshal args to JSON for the Bellman tool
		jsonArgs, err := json.Marshal(jsArgs)
		if err != nil {
			return j.runtime.NewGoError(err)
		}

		// execute the actual go tool
		ctx := j.ctx
		if ctx == nil {
			ctx = context.Background()
		}
		res, err := tool.Function(ctx, tools.Call{
			Argument: jsonArgs,
		})
		if err != nil {
			// return error string directly so the LLM can self-correct, e.g., "json: cannot unmarshal number..."
			return j.runtime.ToValue(map[string]any{"ok": false, "error": err.Error()})
		}

		// unmarshal result back to runtime object if possible
		var parsed interface{}
		if err := json.Unmarshal([]byte(res), &parsed); err == nil {
			return j.runtime.ToValue(parsed)
		}

		// otherwise return raw string
		return j.runtime.ToValue(res)
	}

	j.Lock()
	defer j.Unlock()

	err := j.runtime.Set(escapedName, wrapper)
	if err != nil {
		return err
	}

	return nil
}

// Execute runs a code script in the runtime, uses same error handling as LLM (runtime errors return string!)
func (j *JavaScript) Execute(ctx context.Context, code string) (resString string, resErr error, err error) {
	code, resErr = j.Guardrail(code)
	if resErr != nil {
		return "", resErr, nil
	}
	j.Lock()
	defer j.Unlock()

	j.output.set = false // reset output

	j.ctx = ctx
	defer func() { j.ctx = nil }()

	// panic recovery
	defer func() {
		if r := recover(); r != nil {
			j.log("error: runtime panic! recovering.")
			resErr = fmt.Errorf("critical runtime panic: %v", r)
			err = nil
		}
	}()

	// timeout and context interrupt
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	stop := context.AfterFunc(ctx, func() {
		j.log("error: runtime interrupted", "error", ctx.Err())
		j.runtime.Interrupt(fmt.Sprintf("execution interrupted: %v", ctx.Err()))
	})
	defer stop()

	_, resErr = j.runtime.RunString(code)
	if resErr != nil {
		j.log("error: runtime error!")
		return "", resErr, nil
	}

	// if result(); used, return the value
	if j.output.set {
		return j.output.value, nil, nil
	}

	return nilValue, nil, nil
}

// Matches anything that IS NOT a letter, number, underscore, or dollar sign
var invalidJSFuncSymbols = regexp.MustCompile(`[^a-zA-Z0-9_$]`)

func escapeFunctionName(name string) string {
	safeName := invalidJSFuncSymbols.ReplaceAllString(name, "_")

	// JS identifiers cannot start with a number
	if len(safeName) > 0 && safeName[0] >= '0' && safeName[0] <= '9' {
		safeName = "_" + safeName
	}

	return safeName
}

// registerReturn registers the custom return function in Goja, that returns the value from the PTC tools code
func (j *JavaScript) registerReturn() (*JavaScript, error) {
	out := &resultOutput{}
	j.output = out

	err := j.runtime.Set(returnFunc, func(call goja.FunctionCall) goja.Value {
		if len(call.Arguments) == 0 {
			return goja.Undefined()
		}
		b, err := json.Marshal(call.Argument(0).Export())
		if err != nil {
			out.value = fmt.Sprintf(`{"error": "Failed to serialize return value: %v."}`, err)
			out.set = true
			return goja.Undefined()
		}
		out.value = string(b)
		out.set = true
		return goja.Undefined()
	})
	if err != nil {
		return nil, err
	}

	return j, nil
}

// Guardrail guardrails code before exec; important since LLMs trained for diff. coding objectives
func (j *JavaScript) Guardrail(code string) (string, error) {
	if code == "" {
		j.log("guardrail empty code")
		return code, errors.New("no javascript code provided. validate tool input arguments, required format: '{\"code\": string}'")
	}

	if strings.Contains(code, "async ") || strings.Contains(code, "await") || strings.Contains(code, "async(") {
		j.log("guardrail async code")
		return code, errors.New("runtime error: async functions are unavailable in this runtime. must use synchronous, blocking calls (e.g., 'var x = tool()')")
	}

	if strings.Contains(code, "console.log(") || strings.Contains(code, "print(") {
		j.log("guardrail console/print usage")
		return code, errors.New("runtime error: console.log() and print() are not for returning data")
	}

	if !strings.Contains(code, fmt.Sprintf("%s(", returnFunc)) {
		j.log("guardrail missing result()")
		return code, errors.New("runtime error: script must call result(value) exactly once to return data. example: result({ a, b })")
	}

	return code, nil
}

// SystemFragment creates the system fragment using template and tools
func (j *JavaScript) SystemFragment(tool ...tools.Tool) (string, error) {
	sigs := functionSignatures(tool...)

	data := TemplateData{
		PTCToolName:    j.toolName,
		Signatures:     sigs,
		ReturnFunction: returnFunc,
	}
	var buf bytes.Buffer
	if err := parsedTemplates.ExecuteTemplate(&buf, "ptc_system_prompt", data); err != nil {
		j.log("failed to execute system prompt template", "error", err)
		return "", err
	}

	return buf.String(), nil
}

func functionSignatures(tool ...tools.Tool) []FunctionSignatureData {
	signatures := make([]FunctionSignatureData, 0, len(tool))
	for _, t := range tool {
		// figure out argument node
		var argNode *TSNode
		if t.ArgumentSchema != nil {
			argNode = SchemaToNode("", t.ArgumentSchema, true, "")
		}

		// figure out return node
		var returnNode *TSNode
		unknownSchema := true
		if t.ResponseSchema != nil {
			returnNode = SchemaToNode("", t.ResponseSchema, true, "")
			// if it is a populated schema, we safely know the shape
			if !(returnNode.Type == "object" && len(returnNode.Properties) == 0) {
				unknownSchema = false
			}
		}

		signatures = append(signatures, FunctionSignatureData{
			Name:          escapeFunctionName(t.Name),
			Description:   t.Description,
			ArgumentNode:  argNode,
			ReturnNode:    returnNode,
			UnknownSchema: unknownSchema,
		})
	}
	return signatures
}

// SchemaToNode recursively converts a map-based schema.JSON into a deterministic TSNode struct tree.
// Note: ONLY data extraction, sorting, and cleaning happens here. NO formatting (except indentation...).
func SchemaToNode(name string, s *schema.JSON, isRequired bool, currentIndent string) *TSNode {
	if s == nil {
		return &TSNode{Name: name, Type: "any", Required: isRequired}
	}

	// Clean the description for template injection
	cleanDesc := strings.TrimSpace(strings.ReplaceAll(s.Description, "\n", " "))

	node := &TSNode{
		Name:        name,
		Required:    isRequired,
		Description: cleanDesc,
		Indent:      currentIndent,
	}

	switch s.Type {
	case "string", "boolean":
		node.Type = string(s.Type)
	case "integer", "number":
		node.Type = "number"
	case "array":
		node.Type = "array"
		if s.Items != nil {
			node.Items = SchemaToNode("", s.Items, true, currentIndent)
		} else {
			node.Items = &TSNode{Type: "any", Indent: currentIndent}
		}
	case "object":
		node.Type = "object"
		if len(s.Properties) > 0 {
			reqMap := make(map[string]bool)
			for _, r := range s.Required {
				reqMap[r] = true
			}

			// must sort keys for deterministic prompt gen
			keys := make([]string, 0, len(s.Properties))
			for k := range s.Properties {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			nextIndent := currentIndent + "  "
			for _, key := range keys {
				propReq := reqMap[key]
				node.Properties = append(node.Properties, SchemaToNode(key, s.Properties[key], propReq, nextIndent))
			}
		}
	default:
		node.Type = "any"
	}

	return node
}

func (j *JavaScript) SetLogger(logger *slog.Logger) *JavaScript {
	j.Log = logger
	return j
}
