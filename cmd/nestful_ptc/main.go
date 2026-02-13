package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/agent"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
)

type ToolSpec struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  map[string]any     `json:"parameters"`
	Outputs     map[string]ToolOut `json:"output_parameters"`
}

type ToolOut struct {
	Type        any    `json:"type,omitempty"`
	Description string `json:"description,omitempty"`
}

type Sample struct {
	SampleID   string     `json:"sample_id"`
	Input      string     `json:"input"`
	Output     any        `json:"output"`
	GoldAnswer any        `json:"gold_answer"`
	Tools      []ToolSpec `json:"tools"`
}

type PredCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Label     string          `json:"label"`
}

type ToolTraceEvent struct {
	Index      int             `json:"index"`
	Name       string          `json:"name"`
	Args       json.RawMessage `json:"args"`
	Ok         bool            `json:"ok"`
	Output     json.RawMessage `json:"output,omitempty"`
	Error      string          `json:"error,omitempty"`
	DurationMs int64           `json:"duration_ms"`
}

type traceCollector struct {
	mu     sync.Mutex
	next   int
	events []ToolTraceEvent
}

func (t *traceCollector) start(name string, args []byte) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	idx := t.next
	t.next++
	var raw json.RawMessage
	_ = json.Unmarshal(args, &raw)
	t.events = append(t.events, ToolTraceEvent{Index: idx, Name: name, Args: raw, Ok: false})
	return idx
}

func (t *traceCollector) finishOK(idx int, durMs int64, out []byte) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.events {
		if t.events[i].Index != idx {
			continue
		}
		t.events[i].Ok = true
		t.events[i].DurationMs = durMs
		var raw json.RawMessage
		if json.Unmarshal(out, &raw) == nil {
			t.events[i].Output = raw
		}
		return
	}
}

func (t *traceCollector) finishErr(idx int, durMs int64, errMsg string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.events {
		if t.events[i].Index != idx {
			continue
		}
		t.events[i].Ok = false
		t.events[i].DurationMs = durMs
		t.events[i].Error = errMsg
		return
	}
}

func (t *traceCollector) Events() []ToolTraceEvent {
	t.mu.Lock()
	defer t.mu.Unlock()
	cp := make([]ToolTraceEvent, len(t.events))
	copy(cp, t.events)
	return cp
}

func main() {
	// Load .env if present (non-fatal). Override path via NESTFUL_ENV.
	_ = godotenv.Load(getenvDefault("NESTFUL_ENV", ".env"))

	var (
		bellmanURL   = flag.String("bellman-url", getenvDefault("BELLMAN_URL", ""), "Bellman base URL, e.g. https://bellman.modularfinance.ai/v1")
		bellmanToken = flag.String("bellman-token", getenvDefault("BELLMAN_TOKEN", ""), "Bellman token")
		bellmanName  = flag.String("bellman-key-name", getenvDefault("BELLMAN_KEY_NAME", "test"), "Bellman key name (auth prefix)")
		modelFQN     = flag.String("model", getenvDefault("NESTFUL_MODEL", getenvDefault("BELLMAN_MODEL", "openai/gpt-4o-mini")), "Model as provider/name")

		datasetPath = flag.String("dataset", getenvDefault("NESTFUL_DATASET", ""), "Path to NESTFUL jsonl dataset")
		execDir     = flag.String("execdir", getenvDefault("NESTFUL_EXECDIR", ""), "Path to executable_functions dir")
		outPath     = flag.String("out", getenvDefault("NESTFUL_OUT", ""), "Output jsonl path")
		pythonBin   = flag.String("python", getenvDefault("NESTFUL_PYTHON", "python"), "Python executable")
		startAt     = flag.Int("start", getenvIntDefault("NESTFUL_START", 0), "Skip first N samples")
		limit       = flag.Int("limit", getenvIntDefault("NESTFUL_LIMIT", 0), "Process at most N samples (0 = all)")

		maxDepth    = flag.Int("max-depth", getenvIntDefault("NESTFUL_MAX_DEPTH", 10), "Agent max depth")
		parallelism = flag.Int("parallelism", getenvIntDefault("NESTFUL_PARALLELISM", 0), "Agent parallelism (0 = default)")
		temperature = flag.Float64("temperature", getenvFloatDefault("NESTFUL_TEMPERATURE", 0.0), "Model temperature")
		maxTokens   = flag.Int("max-tokens", getenvIntDefault("NESTFUL_MAX_TOKENS", 1000), "Max output tokens")
		usePTC      = flag.Bool("use-ptc", getenvBoolDefault("NESTFUL_USE_PTC", true), "If true, run tools in PTC (code_execution); if false, expose tools directly")
	)
	flag.Parse()

	if strings.TrimSpace(*bellmanURL) == "" {
		exitf("--bellman-url (or BELLMAN_URL) is required")
	}
	if strings.TrimSpace(*bellmanToken) == "" {
		exitf("--bellman-token (or BELLMAN_TOKEN) is required")
	}
	if strings.TrimSpace(*datasetPath) == "" {
		exitf("--dataset is required")
	}
	if strings.TrimSpace(*execDir) == "" {
		exitf("--execdir is required")
	}
	if strings.TrimSpace(*outPath) == "" {
		exitf("--out is required")
	}

	model, err := parseModel(*modelFQN)
	if err != nil {
		exitf("invalid --model: %v", err)
	}

	if _, err := os.Stat(*datasetPath); err != nil {
		exitf("cannot access dataset: %v", err)
	}
	if st, err := os.Stat(*execDir); err != nil {
		exitf("cannot access execdir: %v", err)
	} else if !st.IsDir() {
		exitf("execdir is not a directory")
	}

	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		exitf("cannot create output dir: %v", err)
	}

	client := bellman.New(*bellmanURL, bellman.Key{Name: *bellmanName, Token: *bellmanToken})

	in, err := os.Open(*datasetPath)
	if err != nil {
		exitf("open dataset: %v", err)
	}
	defer in.Close()

	out, err := os.Create(*outPath)
	if err != nil {
		exitf("create out: %v", err)
	}
	defer out.Close()

	scanner := bufio.NewScanner(in)
	// Allow long lines.
	scanner.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)

	lineNo := 0
	processed := 0
	skipped := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if *startAt > 0 && skipped < *startAt {
			skipped++
			continue
		}
		if *limit > 0 && processed >= *limit {
			break
		}

		var s Sample
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			fmt.Fprintf(os.Stderr, "[line %d] skip: invalid json: %v\n", lineNo, err)
			continue
		}

		processed++
		sampleStart := time.Now()
		fmt.Fprintf(os.Stderr, "[%d] sample_id=%s tools=%d\n", processed, s.SampleID, len(s.Tools))

		collector := &traceCollector{}
		bellmanTools := buildTools(s.Tools, *execDir, *pythonBin, collector, *usePTC)

		systemPrompt := "You are an assistant. Use the provided tools via the code_execution environment to compute the answer. " +
			"Do not guess. Prefer tool calls for calculations and data transformations. Keep the final answer short."

		llm := client.Generator().
			Model(model).
			System(systemPrompt).
			WithContext(context.Background()).
			SetTools(bellmanTools...).
			SetPTCLanguage(tools.JavaScript).
			Temperature(*temperature).
			MaxTokens(*maxTokens)

		res, runErr := agent.Run[string](*maxDepth, *parallelism, llm, prompt.AsUser(s.Input))
		if runErr != nil {
			fmt.Fprintf(os.Stderr, "[%d] agent error: %v\n", processed, runErr)
		} else {
			fmt.Fprintf(os.Stderr, "[%d] ok depth=%d tokens=%d duration=%s\n", processed, res.Depth, res.Metadata.TotalTokens, time.Since(sampleStart).Truncate(time.Millisecond))
		}

		events := collector.Events()
		predCalls := make([]PredCall, 0, len(events))
		for i, e := range events {
			predCalls = append(predCalls, PredCall{
				Name:      e.Name,
				Arguments: e.Args,
				Label:     fmt.Sprintf("$var_%d", i+1),
			})
		}
		predJSON := mustJSON(predCalls)

		outItem := map[string]any{
			"sample_id":      s.SampleID,
			"input":          s.Input,
			"generated_text": string(predJSON),
			"output":         string(mustJSON(s.Output)),
			"gold_answer":    string(mustJSON(s.GoldAnswer)),
			"tools":          string(mustJSON(s.Tools)),
			"ptc_trace":      events,
			"ptc_ok":         runErr == nil,
		}

		if _, err := out.Write(append(mustJSON(outItem), '\n')); err != nil {
			exitf("write out: %v", err)
		}
	}
	if err := scanner.Err(); err != nil {
		exitf("scan dataset: %v", err)
	}

	fmt.Printf("Wrote: %s\n", *outPath)
}

func parseModel(fqn string) (gen.Model, error) {
	fqn = strings.TrimSpace(fqn)
	provider, name, found := strings.Cut(fqn, "/")
	if !found {
		provider, name, found = strings.Cut(fqn, ".")
	}
	if !found || strings.TrimSpace(provider) == "" || strings.TrimSpace(name) == "" {
		return gen.Model{}, fmt.Errorf("expected provider/name (or provider.name), got %q", fqn)
	}
	provider = canonicalProvider(provider)
	name = canonicalModelName(name)
	return gen.Model{Provider: provider, Name: name}, nil
}

func canonicalProvider(p string) string {
	pl := strings.ToLower(strings.TrimSpace(p))
	switch pl {
	case "openai":
		return "OpenAI"
	case "vertexai", "vertex":
		return "VertexAI"
	case "anthropic":
		return "Anthropic"
	case "ollama":
		return "Ollama"
	case "vllm":
		return "vLLM"
	case "voyageai", "voyage":
		return "VoyageAI"
	default:
		// Fall back to given provider; note that provider names are case-sensitive in bellmand.
		return strings.TrimSpace(p)
	}
}

func canonicalModelName(n string) string {
	n = strings.TrimSpace(n)
	if n == "" {
		return n
	}
	// Common convenience: allow snake_case and missing dash in gpt-4o.*
	n = strings.ReplaceAll(n, "_", "-")
	if strings.HasPrefix(n, "gpt4o-") {
		n = "gpt-4o-" + strings.TrimPrefix(n, "gpt4o-")
	}
	return n
}

func buildTools(specs []ToolSpec, execDir string, pythonBin string, collector *traceCollector, enablePTC bool) []tools.Tool {
	out := make([]tools.Tool, 0, len(specs))
	for _, s := range specs {
		spec := s
		outKeys := sortedKeys(spec.Outputs)

		desc := strings.TrimSpace(spec.Description)
		if len(outKeys) > 0 {
			desc = strings.TrimSpace(desc + " Returns JSON with keys: " + strings.Join(outKeys, ", "))
		}

		argSchema := &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}
		var required []string
		for pname, pdef := range spec.Parameters {
			ps := schemaFromAny(pdef)
			if ps == nil {
				ps = &schema.JSON{}
			}
			argSchema.Properties[pname] = ps
			if isRequired(pdef) {
				required = append(required, pname)
			}
		}
		if len(required) > 0 {
			sort.Strings(required)
			argSchema.Required = required
		}

		t := tools.Tool{
			Name:           spec.Name,
			Description:    desc,
			ArgumentSchema: argSchema,
			UsePTC:         enablePTC,
		}
		t.Function = func(ctx context.Context, call tools.Call) (string, error) {
			if ctx == nil {
				ctx = context.Background()
			}
			idx := collector.start(spec.Name, call.Argument)
			start := time.Now()
			outStr, err := execNestfulPython(ctx, pythonBin, execDir, spec.Name, call.Argument, outKeys)
			dur := time.Since(start).Milliseconds()
			if err != nil {
				collector.finishErr(idx, dur, err.Error())
				// Do not abort agent loop; surface tool error as JSON.
				return string(mustJSON(map[string]any{"error": err.Error()})), nil
			}
			collector.finishOK(idx, dur, []byte(outStr))
			return outStr, nil
		}

		out = append(out, t)
	}
	return out
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
		// Sometimes params are bare types.
		if s, ok := v.(string); ok {
			return schemaFromTypeString(s)
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

	// If it is already JSON-schema-like, try to map common fields.
	if typ, ok := m["type"]; ok {
		switch t := typ.(type) {
		case string:
			applyTypeFromString(js, t)
		case []any:
			// anyOf style: if includes string+number, fall back to no type.
			// Keep it permissive.
			_ = t
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

func schemaFromTypeString(s string) *schema.JSON {
	js := &schema.JSON{}
	applyTypeFromString(js, s)
	return js
}

func applyTypeFromString(js *schema.JSON, t string) {
	ls := strings.ToLower(strings.TrimSpace(t))
	switch {
	case ls == "integer" || ls == "int":
		js.Type = schema.Integer
	case ls == "number" || ls == "float":
		js.Type = schema.Number
	case ls == "string":
		js.Type = schema.String
	case ls == "boolean" || ls == "bool":
		js.Type = schema.Boolean
	case ls == "array" || strings.Contains(ls, "list"):
		js.Type = schema.Array
		js.Items = &schema.JSON{}
	case ls == "object" || ls == "dict" || strings.Contains(ls, "map"):
		js.Type = schema.Object
	case strings.Contains(ls, "int") || strings.Contains(ls, "float") || strings.Contains(ls, "number"):
		// e.g. "int or float"
		js.Type = schema.Number
	default:
		// keep permissive
	}
}

func sortedKeys[V any](m map[string]V) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func execNestfulPython(ctx context.Context, pythonBin string, execDir string, toolName string, argsJSON []byte, outputKeys []string) (string, error) {
	if strings.TrimSpace(pythonBin) == "" {
		pythonBin = "python"
	}
	// Keep the python runner self-contained.
	py := strings.Join([]string{
		"import os, sys, json, importlib.util",
		"tool = os.environ.get('NESTFUL_TOOL_NAME','')",
		"exec_dir = os.environ.get('NESTFUL_EXEC_DIR','')",
		"output_keys = json.loads(os.environ.get('NESTFUL_OUTPUT_KEYS_JSON','[]'))",
		"args = json.load(sys.stdin)",
		"func_map_path = os.path.join(exec_dir, 'func_file_map.json')",
		"file_name = None",
		"if os.path.exists(func_map_path):",
		"  with open(func_map_path, 'r', encoding='utf-8') as f:",
		"    m = json.load(f)",
		"  file_name = m.get(tool)",
		"if not file_name:",
		"  file_name = 'basic_functions.py'",
		"file_path = os.path.join(exec_dir, file_name)",
		"spec = importlib.util.spec_from_file_location('nestful_exec_mod', file_path)",
		"mod = importlib.util.module_from_spec(spec)",
		"spec.loader.exec_module(mod)",
		"if not hasattr(mod, tool):",
		"  raise Exception(f'function not found: {tool} in {file_name}')",
		"fn = getattr(mod, tool)",
		"res = None",
		"try:",
		"  if isinstance(args, dict):",
		"    res = fn(**args)",
		"  else:",
		"    res = fn(args)",
		"except TypeError:",
		"  # Fallback: positional by arg_<n> ordering if present.",
		"  if isinstance(args, dict):",
		"    def _arg_i(k):",
		"      if k.startswith('arg_'):",
		"        try: return int(k.split('_',1)[1])",
		"        except: return 10**9",
		"      return 10**9",
		"    keys = sorted(list(args.keys()), key=_arg_i)",
		"    res = fn(*[args[k] for k in keys])",
		"  else:",
		"    res = fn(args)",
		"out = None",
		"if isinstance(res, dict):",
		"  out = res",
		"elif len(output_keys) == 1:",
		"  out = { output_keys[0]: res }",
		"elif isinstance(res, (list, tuple)) and len(output_keys) == len(res):",
		"  out = { k: v for k, v in zip(output_keys, res) }",
		"else:",
		"  out = { 'result': res }",
		"sys.stdout.write(json.dumps(out))",
	}, "\n")

	cmd := exec.CommandContext(ctx, pythonBin, "-c", py)
	cmd.Env = append(os.Environ(),
		"NESTFUL_TOOL_NAME="+toolName,
		"NESTFUL_EXEC_DIR="+execDir,
		"NESTFUL_OUTPUT_KEYS_JSON="+string(mustJSON(outputKeys)),
	)
	cmd.Stdin = bytes.NewReader(argsJSON)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", errors.New(msg)
	}

	out := strings.TrimSpace(stdout.String())
	if out == "" {
		return "", errors.New("empty tool output")
	}
	var tmp any
	if json.Unmarshal([]byte(out), &tmp) != nil {
		return "", fmt.Errorf("tool output is not valid json: %s", out)
	}
	return out, nil
}

func getenvDefault(key string, def string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	return v
}

func getenvIntDefault(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	var out int
	if _, err := fmt.Sscanf(v, "%d", &out); err != nil {
		return def
	}
	return out
}

func getenvFloatDefault(key string, def float64) float64 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	var out float64
	if _, err := fmt.Sscanf(v, "%f", &out); err != nil {
		return def
	}
	return out
}

func getenvBoolDefault(key string, def bool) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if v == "" {
		return def
	}
	switch v {
	case "1", "true", "t", "yes", "y", "on":
		return true
	case "0", "false", "f", "no", "n", "off":
		return false
	default:
		return def
	}
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
