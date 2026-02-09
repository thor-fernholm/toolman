package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/agent"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/services/vertexai"
	"github.com/modfin/bellman/tools"
)

// This runner reads StableToolBench solvable_queries JSON (per group) and runs Bellman with PTC enabled.
// Tools are created per query from api_list and executed against a /virtual endpoint (cache-replay server).

type queryFileEntry struct {
	QueryID int             `json:"query_id"`
	Query   string          `json:"query"`
	APIList []apiListRecord `json:"api_list"`
}

type apiListRecord struct {
	CategoryName      string           `json:"category_name"`
	ToolName          string           `json:"tool_name"`
	APIName           string           `json:"api_name"`
	APIDescription    string           `json:"api_description"`
	RequiredParams    []apiParamRecord `json:"required_parameters"`
	OptionalParams    []apiParamRecord `json:"optional_parameters"`
	HTTPMethod        string           `json:"method"`
	TemplateResponse  any              `json:"template_response"`
	TemplateResponse2 any              `json:"template_response_2"`
}

type apiParamRecord struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Default     any    `json:"default"`
}

type virtualReq struct {
	Category     string `json:"category"`
	ToolName     string `json:"tool_name"`
	APIName      string `json:"api_name"`
	ToolInput    any    `json:"tool_input"`
	Strip        string `json:"strip"`
	ToolbenchKey string `json:"toolbench_key"`
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		v := strings.TrimSpace(os.Getenv(k))
		if v != "" {
			return v
		}
	}
	return ""
}

func callVirtual(ctx context.Context, virtualURL string, req virtualReq) (string, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, virtualURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	hreq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(hreq)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	b, err := io.ReadAll(res.Body)
	if err != nil {
		return "", err
	}
	if res.StatusCode != http.StatusOK {
		return "", fmt.Errorf("/virtual status %d: %s", res.StatusCode, string(b))
	}
	return string(b), nil
}

var nonIdent = regexp.MustCompile(`[^a-zA-Z0-9_]+`)

func standardizeIdent(s string) string {
	//fix so every JS function names are written correct
	s = nonIdent.ReplaceAllString(s, "_")
	s = strings.ToLower(s)
	s = regexp.MustCompile(`_+`).ReplaceAllString(s, "_")
	s = strings.Trim(s, "_")
	if s == "" {
		return "tool"
	}
	if s[0] >= '0' && s[0] <= '9' {
		s = "get_" + s
	}
	switch s {
	case "from", "class", "return", "false", "true", "id", "and":
		s = "is_" + s
	}
	return s
}

func schemaTypeFromStableToolBench(t string) schema.JSONType {
	// StableToolBench uses "string", "int", "float" and sometimes others.
	switch strings.ToLower(strings.TrimSpace(t)) {
	case "string", "str", "text":
		return schema.String
	case "int", "integer":
		return schema.Integer
	case "float", "double", "number":
		return schema.Number
	case "bool", "boolean":
		return schema.Boolean
	default:
		return schema.String
	}
}

func buildArgSchema(required, optional []apiParamRecord) *schema.JSON {
	props := map[string]*schema.JSON{}
	reqNames := make([]string, 0, len(required))

	add := func(p apiParamRecord, required bool) {
		name := standardizeIdent(p.Name)
		if name == "" {
			return
		}
		props[name] = &schema.JSON{
			Type:        schemaTypeFromStableToolBench(p.Type),
			Description: strings.TrimSpace(p.Description),
		}
		if required {
			reqNames = append(reqNames, name)
		}
	}

	for _, p := range required {
		add(p, true)
	}
	for _, p := range optional {
		add(p, false)
	}
	sort.Strings(reqNames)

	return &schema.JSON{
		Type:       schema.Object,
		Properties: props,
		Required:   reqNames,
	}
}

func newAPITool(rec apiListRecord, virtualURL, toolbenchKey string) tools.Tool {
	// Name must be a JS-friendly identifier for PTC (code_execution will expose these as JS functions).
	fnName := standardizeIdent(rec.APIName) + "_for_" + standardizeIdent(rec.ToolName)

	descParts := []string{}
	if strings.TrimSpace(rec.APIDescription) != "" {
		descParts = append(descParts, strings.TrimSpace(rec.APIDescription))
	}
	if strings.TrimSpace(rec.HTTPMethod) != "" {
		descParts = append(descParts, "method: "+strings.TrimSpace(rec.HTTPMethod))
	}
	desc := strings.Join(descParts, "; ")
	if desc == "" {
		desc = "StableToolBench API wrapper"
	}

	argSchema := buildArgSchema(rec.RequiredParams, rec.OptionalParams)

	t := tools.NewTool(fnName,
		tools.WithDescription(desc),
		tools.WithPTC(false),
		tools.WithFunction(func(ctx context.Context, call tools.Call) (string, error) {
			// call.Argument is JSON of the argument object
			var args map[string]any
			if len(call.Argument) > 0 {
				if err := json.Unmarshal(call.Argument, &args); err != nil {
					return "", err
				}
			} else {
				args = map[string]any{}
			}

			if ctx == nil {
				ctx = context.Background()
			}
			return callVirtual(ctx, virtualURL, virtualReq{
				Category:     rec.CategoryName,
				ToolName:     rec.ToolName,
				APIName:      rec.APIName,
				ToolInput:    args,
				Strip:        "",
				ToolbenchKey: toolbenchKey,
			})
		}),
	)
	// attach schema (cannot use tools.WithArgSchema with dynamic schema)
	t.ArgumentSchema = argSchema
	return t
}

type openAIFunctionSpec struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]any         `json:"parameters"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

func schemaToOpenAIParams(s *schema.JSON) map[string]any {
	// StableToolBench wants a schema in OpenAi style, so have to convert.
	if s == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}

	props := map[string]any{}
	keys := make([]string, 0, len(s.Properties))
	for k := range s.Properties {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		p := s.Properties[k]
		props[k] = map[string]any{
			"type":        string(p.Type),
			"description": p.Description,
		}
	}

	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   s.Required,
	}
}

func promptsToToolbenchConversation(systemPrompt, userQuery string, toolPrompts []prompt.Prompt, finalAnswer string) []map[string]any {
	conv := []map[string]any{
		{"role": "system", "content": systemPrompt},
		{"role": "user", "content": userQuery},
	}

	// Some providers don't populate tool call IDs. StableToolBench expects tool_call_id to match.
	autoID := 0
	lastAutoID := ""

	for _, p := range toolPrompts {
		switch p.Role {
		case prompt.ToolCallRole:
			if p.ToolCall == nil {
				continue
			}
			id := p.ToolCall.ToolCallID
			if id == "" {
				autoID++
				id = fmt.Sprintf("call_%d", autoID)
				lastAutoID = id
			}
			conv = append(conv, map[string]any{
				"role": "assistant",
				"tool_calls": []map[string]any{
					{
						"id":   id,
						"type": "function",
						"function": map[string]any{
							"name":      p.ToolCall.Name,
							"arguments": string(p.ToolCall.Arguments),
						},
					},
				},
			})
		case prompt.ToolResponseRole:
			if p.ToolResponse == nil {
				continue
			}
			id := p.ToolResponse.ToolCallID
			if id == "" {
				if lastAutoID != "" {
					id = lastAutoID
				} else {
					autoID++
					id = fmt.Sprintf("call_%d", autoID)
				}
			}
			conv = append(conv, map[string]any{
				"role":         "tool",
				"tool_call_id": id,
				"content":      p.ToolResponse.Response,
			})
		default:
			// ignore user/assistant text prompts here; we record only tool call/resp
		}
	}
	/*
		conv = append(conv, map[string]any{
			"role":    "assistant",
			"content": finalAnswer,
		})
	*/
	finishArgs, _ := json.Marshal(map[string]any{
		"final_answer": finalAnswer,
	})

	autoID++
	finishID := fmt.Sprintf("call_%d", autoID)

	conv = append(conv, map[string]any{
		"role": "assistant",
		"tool_calls": []map[string]any{
			{
				"id":   finishID,
				"type": "function",
				"function": map[string]any{
					"name":      "Finish",
					"arguments": string(finishArgs),
				},
			},
		},
	})

	conv = append(conv, map[string]any{
		"role":         "tool",
		"tool_call_id": finishID,
		"content":      "",
	})

	return conv

}

func prettyJSON(b []byte) string {
	if len(b) == 0 {
		return "{}"
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return string(b)
	}
	out, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return string(b)
	}
	return string(out)
}

func writeReadableRun(outDir string, qid int, method string, systemPrompt string, userQuery string, toolPrompts []prompt.Prompt, finalAnswer string) error {
	var sb strings.Builder

	sb.WriteString("== System Prompt ==\n")
	sb.WriteString(systemPrompt)
	sb.WriteString("\n\n")

	sb.WriteString("== User Query ==\n")
	sb.WriteString(userQuery)
	sb.WriteString("\n\n")

	sb.WriteString("== Tool Calls / Responses ==\n")
	for _, p := range toolPrompts {
		switch p.Role {
		case prompt.ToolCallRole:
			if p.ToolCall == nil {
				continue
			}
			sb.WriteString("-- TOOL CALL --\n")
			sb.WriteString(fmt.Sprintf("name: %s\n", p.ToolCall.Name))
			if p.ToolCall.ToolCallID != "" {
				sb.WriteString(fmt.Sprintf("id: %s\n", p.ToolCall.ToolCallID))
			}
			sb.WriteString("arguments:\n")
			sb.WriteString(prettyJSON(p.ToolCall.Arguments))
			sb.WriteString("\n\n")

		case prompt.ToolResponseRole:
			if p.ToolResponse == nil {
				continue
			}
			sb.WriteString("-- TOOL RESPONSE --\n")
			sb.WriteString(fmt.Sprintf("name: %s\n", p.ToolResponse.Name))
			if p.ToolResponse.ToolCallID != "" {
				sb.WriteString(fmt.Sprintf("tool_call_id: %s\n", p.ToolResponse.ToolCallID))
			}
			sb.WriteString("content:\n")
			sb.WriteString(p.ToolResponse.Response)
			sb.WriteString("\n\n")
		}
	}

	sb.WriteString("== Final Answer ==\n")
	sb.WriteString(finalAnswer)
	sb.WriteString("\n")

	// Example filename: 588_PTC@1_readable.txt
	p := filepath.Join(outDir, fmt.Sprintf("%d_%s_readable.txt", qid, method))
	return os.WriteFile(p, []byte(sb.String()), 0o644)
}

func main() {
	_ = godotenv.Load(".env")
	_ = godotenv.Load("../../.env")
	var (
		queriesPath    = flag.String("queries", "", "Path to StableToolBench group JSON (e.g. solvable_queries/test_instruction/G1_instruction.json)")
		outDir         = flag.String("out", "data/answer/virtual_myptc", "Output directory")
		method         = flag.String("method", "PTC@1", "Method name used in output filenames")
		modelFQN       = flag.String("model", "", "Model FQN, e.g. 'ollama/llama3.1' or 'openai/gpt-4o-mini'")
		bellmanURL     = flag.String("bellman-url", os.Getenv("BELLMAN_URL"), "Bellman proxy base URL (optional; set to use proxy)")  //flag.String("bellman-url", os.Getenv("BELLMAN_URL"), "Bellman proxy base URL (optional; set to use proxy)")
		bellmanToken   = flag.String("bellman-token", os.Getenv("BELLMAN_TOKEN"), "Bellman proxy token (optional; set to use proxy)") //flag.String("bellman-token", os.Getenv("BELLMAN_TOKEN"), "Bellman proxy token (optional; set to use proxy)")
		googleProject  = flag.String("google-project", firstEnv("GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT", "GCLOUD_PROJECT"), "GCP project id (or set GOOGLE_CLOUD_PROJECT)")
		googleRegion   = flag.String("google-region", firstEnv("GOOGLE_CLOUD_REGION", "CLOUDSDK_COMPUTE_REGION"), "GCP region (or set GOOGLE_CLOUD_REGION). Use 'global' to reduce 429s")
		googleCredFile = flag.String("google-credential-file", os.Getenv("GOOGLE_APPLICATION_CREDENTIALS"), "Service account JSON file path (optional; default uses ADC)")
		virtualURL     = flag.String("virtual-url", os.Getenv("STB_VIRTUAL_URL"), "StableToolBench /virtual URL (or set STB_VIRTUAL_URL)")
		toolbenchKey   = flag.String("toolbench-key", os.Getenv("TOOLBENCH_KEY"), "ToolBench key (optional; forwarded to /virtual)")
		maxDepth       = flag.Int("max-depth", 10, "Max agent steps")
		parallelism    = flag.Int("parallelism", 0, "Tool execution parallelism (0/1 for sequential)")
		limit          = flag.Int("limit", 0, "Limit number of queries (0 = all)")
		offset         = flag.Int("offset", 0, "Offset into query list")
		sysPrompt      = flag.String("system", "You are a helpful assistant.", "Base system prompt")
	)

	flag.Parse()

	if *queriesPath == "" {
		fmt.Fprintln(os.Stderr, "--queries is required")
		os.Exit(2)
	}
	if *modelFQN == "" {
		fmt.Fprintln(os.Stderr, "--model is required (provider/model)")
		os.Exit(2)
	}
	if *virtualURL == "" {
		fmt.Fprintln(os.Stderr, "virtual URL missing: set --virtual-url or STB_VIRTUAL_URL")
		os.Exit(2)
	}

	model, err := gen.ToModel(*modelFQN)
	if err != nil {
		fmt.Fprintln(os.Stderr, "invalid --model:", err)
		os.Exit(2)
	}

	useProxy := strings.TrimSpace(*bellmanURL) != "" || strings.TrimSpace(*bellmanToken) != ""
	if useProxy {
		if strings.TrimSpace(*bellmanURL) == "" || strings.TrimSpace(*bellmanToken) == "" {
			fmt.Fprintln(os.Stderr, "to use bellman proxy you must set BOTH --bellman-url and --bellman-token (or BELLMAN_URL/BELLMAN_TOKEN)")
			os.Exit(2)
		}
	} else {
		// Direct mode (no proxy) currently supports VertexAI only.
		if model.Provider != vertexai.Provider {
			fmt.Fprintln(os.Stderr, "direct mode supports VertexAI only; set --model to 'VertexAI/<model>' or provide --bellman-url/--bellman-token")
			os.Exit(2)
		}
		if *googleProject == "" {
			fmt.Fprintln(os.Stderr, "missing GCP project: set --google-project or GOOGLE_CLOUD_PROJECT")
			os.Exit(2)
		}
		if *googleRegion == "" {
			fmt.Fprintln(os.Stderr, "missing GCP region: set --google-region or GOOGLE_CLOUD_REGION (try 'global')")
			os.Exit(2)
		}
	}

	raw, err := os.ReadFile(*queriesPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read queries:", err)
		os.Exit(1)
	}

	var entries []queryFileEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		fmt.Fprintln(os.Stderr, "parse queries json:", err)
		os.Exit(1)
	}

	// slice by offset/limit
	if *offset > 0 {
		if *offset >= len(entries) {
			fmt.Fprintln(os.Stderr, "offset beyond entries")
			os.Exit(2)
		}
		entries = entries[*offset:]
	}
	if *limit > 0 && *limit < len(entries) {
		entries = entries[:*limit]
	}

	groupName := strings.TrimSuffix(filepath.Base(*queriesPath), filepath.Ext(*queriesPath))
	groupOutDir := filepath.Join(*outDir, groupName)
	if err := os.MkdirAll(groupOutDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir out:", err)
		os.Exit(1)
	}

	var proxy *bellman.Bellman
	var vertex *vertexai.Google
	if useProxy {
		proxy = bellman.New(*bellmanURL, bellman.Key{Name: "stb", Token: *bellmanToken})
	} else {
		cred := ""
		if strings.TrimSpace(*googleCredFile) != "" {
			b, err := os.ReadFile(*googleCredFile)
			if err != nil {
				fmt.Fprintln(os.Stderr, "read google credential file:", err)
				os.Exit(1)
			}
			cred = string(b)
		}
		v, err := vertexai.New(vertexai.GoogleConfig{
			Project:    *googleProject,
			Region:     *googleRegion,
			Credential: cred,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "vertexai init:", err)
			os.Exit(1)
		}
		vertex = v
	}

	for idx, q := range entries {
		start := time.Now()

		// build tools per query
		queryTools := make([]tools.Tool, 0, len(q.APIList))
		fnSpecs := make([]openAIFunctionSpec, 0, len(q.APIList))
		for _, api := range q.APIList {
			t := newAPITool(api, *virtualURL, *toolbenchKey)
			queryTools = append(queryTools, t)
			fnSpecs = append(fnSpecs, openAIFunctionSpec{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  schemaToOpenAIParams(t.ArgumentSchema),
				Metadata: map[string]interface{}{
					"category":  api.CategoryName,
					"tool_name": api.ToolName,
					"api_name":  api.APIName,
				},
			})
		}

		var g *gen.Generator
		if useProxy {
			g = proxy.Generator().
				Model(model).
				System(*sysPrompt).
				SetTools(queryTools...).
				Temperature(0).
				SetPTCLanguage(tools.JavaScript)
		} else {
			g = vertex.Generator().
				Model(model).
				System(*sysPrompt).
				SetTools(queryTools...).
				Temperature(0).
				SetPTCLanguage(tools.JavaScript)
		}

		res, runErr := agent.Run[string](*maxDepth, *parallelism, g, prompt.AsUser(q.Query))
		final := ""
		toolPrompts := []prompt.Prompt{}
		if res != nil {
			final = res.Result
			toolPrompts = res.Prompts
		}
		if runErr != nil {
			final = "ERROR: " + runErr.Error()
		}

		conv := promptsToToolbenchConversation(*sysPrompt, q.Query, toolPrompts, final)
		fileObj := map[string]any{
			"answer_generation": map[string]any{
				"valid_data":     true,
				"query":          q.Query,
				"function":       fnSpecs,
				"train_messages": []any{conv},
				"final_answer":   final,
			},
		}

		outPath := filepath.Join(groupOutDir, fmt.Sprintf("%d_%s.json", q.QueryID, *method))
		b, _ := json.MarshalIndent(fileObj, "", "  ")
		if err := os.WriteFile(outPath, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			os.Exit(1)
		}

		if err := writeReadableRun(groupOutDir, q.QueryID, *method, *sysPrompt, q.Query, toolPrompts, final); err != nil {
			fmt.Fprintln(os.Stderr, "write readable:", err)
			os.Exit(1)
		}

		dur := time.Since(start)
		fmt.Printf("[%s] %d/%d qid=%d tools=%d time=%s err=%v\n", groupName, idx+1, len(entries), q.QueryID, len(queryTools), dur.Round(time.Millisecond), runErr)
	}
}
