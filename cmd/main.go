package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync/atomic"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/tools"
	"github.com/modfin/bellman/tools/ptc/bfcl"
)

type BenchmarkRequest struct {
	Model          string          `json:"bellman_model"`
	Messages       []Message       `json:"messages"`
	ToolmanHistory []prompt.Prompt `json:"toolman_history"`
	Tools          []interface{}   `json:"tools"`
	Temperature    float64         `json:"temperature"`
	SystemPrompt   string          `json:"system_prompt"`
	EnablePTC      bool            `json:"enable_ptc"`
}

type Message struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	ToolName string `json:"tool_name"`
	ToolID   string `json:"tool_call_id"`
}

type BenchmarkResponse struct {
	ToolCalls      []bfcl.ExtractedCall `json:"tool_calls"`
	ToolCallIDs    []string             `json:"tool_call_ids"`
	ToolmanHistory []prompt.Prompt      `json:"toolman_history"`
	Content        string               `json:"content"`       // Any thought/text
	InputTokens    int                  `json:"input_tokens"`  // Added for tracking
	OutputTokens   int                  `json:"output_tokens"` // Added for tracking
}

var (
	GlobalInputTokens  uint64
	GlobalOutputTokens uint64
)

func main() {
	// godotenv.Load() ...
	err := godotenv.Load()
	if err != nil {
		panic(err)
	}

	//http.HandleFunc("/bfcl", handleGenerateBFCL)

	// Register API Endpoint
	http.HandleFunc("/bfcl", bfcl.MiddlewareDebugLogger(handleGenerateBFCL))

	// Register Debug UI Endpoints
	http.HandleFunc("/debug", bfcl.HandleDebugUI)
	http.HandleFunc("/debug/api/data", bfcl.HandleDebugData)
	http.HandleFunc("/debug/api/clear", bfcl.HandleDebugClear)

	fmt.Println("---------------------------------------------------------")
	fmt.Println(" Toolman Server Running")
	fmt.Println(" API Endpoint:   http://localhost:8080/bfcl")
	fmt.Println(" Debug UI:       http://localhost:8080/debug")
	fmt.Println("---------------------------------------------------------")

	fmt.Println("Toolman Benchmark Server running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleGenerateBFCL(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	//PrintRequest(r) // Debug requests

	var req BenchmarkRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	client := bellman.New(bellmanUrl, bellman.Key{Name: "bfcl", Token: bellmanToken})

	bfclTools := bfcl.ParseJsonSchemaTools(req.Tools, req.EnablePTC)

	toolmanHistory := req.ToolmanHistory

	// count toolman user messages
	toolmanUserCount := 0
	for _, p := range toolmanHistory {
		if p.Role == prompt.UserRole {
			toolmanUserCount++
		}
	}
	// add trailing messages from BFCL
	bfclUserCount := 0
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			// only add new user messages from bfcl (not in toolman hist.)
			bfclUserCount++
			if bfclUserCount > toolmanUserCount {
				toolmanHistory = append(toolmanHistory, prompt.AsUser(m.Content))
			}
		}
	}

	// Add tool response after call!
	var rebuiltHistory []prompt.Prompt
	for i, p := range toolmanHistory {
		switch p.Role {
		case prompt.ToolCallRole:
			rebuiltHistory = append(rebuiltHistory, p)
			// find all corresponding tool results and concatenate
			var concatenatedReturns []string
			for j := 0; j < len(req.Messages); j++ {
				if req.Messages[j].Role == "tool_response" && req.Messages[j].ToolID == p.ToolCall.ToolCallID {
					concatenatedReturns = append(concatenatedReturns, fmt.Sprintf("Function '%s' result: %s.", req.Messages[j].ToolName, req.Messages[j].Content))
				}
			}
			// add JS runtime errors to tool response
			if len(toolmanHistory) > i+1 {
				nextPrompt := toolmanHistory[i+1]
				if nextPrompt.Role == prompt.ToolResponseRole && nextPrompt.ToolResponse.ToolCallID == p.ToolCall.ToolCallID {
					concatenatedReturns = append(concatenatedReturns, nextPrompt.ToolResponse.Response)
				}
			}
			rebuiltHistory = append(rebuiltHistory, prompt.AsToolResponse(p.ToolCall.ToolCallID, p.ToolCall.Name, strings.Join(concatenatedReturns, "\n")))
		case prompt.UserRole:
			rebuiltHistory = append(rebuiltHistory, p)
			//case prompt.AssistantRole: // <-- assistant should only come from toolman response, not added here!
			//	rebuiltHistory = append(rebuiltHistory, p)
		}
	}

	model, err := gen.ToModel(req.Model)
	if err != nil {
		log.Fatalf("error: %e", err)
	}
	//model = openai.GenModel_gpt4_1_mini_250414

	// remove bfcl prompt for PTC - misleading!
	if req.EnablePTC {
		req.SystemPrompt = ""
	}

	llm := client.Generator().Model(model).
		System(req.SystemPrompt).
		SetTools(bfclTools...).
		SetPTCLanguage(tools.JavaScript).
		Temperature(req.Temperature)

	res, err := llm.Prompt(rebuiltHistory...)
	if err != nil {
		log.Printf("Prompt Error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// EXTRACT TOKENS & UPDATE GLOBAL COUNTERS
	inputTokens := res.Metadata.InputTokens
	outputTokens := res.Metadata.OutputTokens

	// Thread-safe increment
	atomic.AddUint64(&GlobalInputTokens, uint64(inputTokens))
	atomic.AddUint64(&GlobalOutputTokens, uint64(outputTokens))

	// Log the running total to the console
	log.Printf("[Token Stats] Request: %d / %d | Global Total: %d / %d",
		inputTokens, outputTokens,
		atomic.LoadUint64(&GlobalInputTokens), atomic.LoadUint64(&GlobalOutputTokens))

	// extract individual new tool calls for bfcl + toolman
	extractedCalls, toolmanCalls, toolCallIDs, err := bfcl.GetToolCalls(res, bfclTools)

	// add new toolman calls to conversation history
	toolmanHistory = append(rebuiltHistory, toolmanCalls...)

	resp := BenchmarkResponse{
		ToolCalls:      extractedCalls,
		ToolCallIDs:    toolCallIDs,
		ToolmanHistory: toolmanHistory,
		Content:        "Tool calls generated", // TODO <-- is this used in bfcl?
		InputTokens:    res.Metadata.InputTokens,
		OutputTokens:   res.Metadata.OutputTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
