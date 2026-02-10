package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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

	http.HandleFunc("/bfcl", handleGenerateBFCL)

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

	//fmt.Printf("\nSystem prompt: %s\n\n", req.SystemPrompt)

	//fmt.Println("Request tools: %v", req.Tools)
	bfclTools := bfcl.ParseJsonSchemaTools(req.Tools, req.EnablePTC)
	//fmt.Printf("\n---------- conversation...\n")
	//for i, m := range req.Messages {
	//	fmt.Printf("msg %v: %v\n", i, m)
	//}

	toolmanHistory := req.ToolmanHistory
	// add trailing messages
	for i, m := range req.Messages {
		switch m.Role {
		//case "tool_response":
		//	// overwrite previous tool with same id (same ptc call)
		//	if messages[len(messages)-1].ToolCall.ToolCallID == m.ToolID {
		//		messages[len(messages)-1] = prompt.AsToolResponse(m.ToolID, "code_execution", m.Content) // Important: this should always be code_execution tool
		//	} else {
		//		messages = append(messages, prompt.AsToolResponse(m.ToolID, "code_execution", m.Content))
		//	}
		case "user":
			// add trailing user message (last message in conversation) // TODO can be multiple...
			if i == len(req.Messages)-1 {
				fmt.Printf("Adding trailing user msg: %+v\n", m)
				toolmanHistory = append(toolmanHistory, prompt.AsUser(m.Content))
			}
			//case "assistant": // don't think this is needed as assistant --> final response?
			//	if i >= len(messages) {
			//		fmt.Printf("Adding trailing assistant msg: %+v\n", m)
			//
			//		messages = append(messages, prompt.AsAssistant(m.Content))
			//	}
		}
	}

	// Add tool response after call!
	var rebuiltHistory []prompt.Prompt
	for _, p := range toolmanHistory {
		switch p.Role {
		case prompt.ToolCallRole:
			rebuiltHistory = append(rebuiltHistory, p)
			// find (last) corresponding tool result
			for j := len(req.Messages) - 1; j >= 0; j-- {
				if req.Messages[j].Role == "tool_response" && req.Messages[j].ToolID == p.ToolCall.ToolCallID {
					rebuiltHistory = append(rebuiltHistory, prompt.AsToolResponse(p.ToolCall.ToolCallID, p.ToolCall.Name, req.Messages[j].Content))
					break
				}
			}
		case prompt.UserRole:
			rebuiltHistory = append(rebuiltHistory, p)
			//case prompt.AssistantRole:
			//	fmt.Printf("!!!!!!!!!!! assistatn role!\n")
			//	rebuiltHistory = append(rebuiltHistory, p)
		}
	}
	//
	//fmt.Printf("\n========== bellman conversation...\n")
	//for i, m := range rebuiltHistory {
	//	fmt.Printf("msg %v: %v\n", i, m)
	//}

	model, err := gen.ToModel(req.Model) // TODO use model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if err != nil {
		log.Fatalf("error: %e", err)
	}
	//model = openai.GenModel_gpt4_1_mini_250414

	llm := client.Generator().Model(model).
		System(req.SystemPrompt). // Use passed system prompt or default
		SetTools(bfclTools...).
		SetPTCLanguage(tools.JavaScript).
		Temperature(req.Temperature)

	res, err := llm.Prompt(rebuiltHistory...)
	if err != nil {
		log.Printf("Prompt Error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// 3. EXTRACT TOKENS & UPDATE GLOBAL COUNTERS
	inputTokens := res.Metadata.InputTokens
	outputTokens := res.Metadata.OutputTokens

	// Thread-safe increment
	atomic.AddUint64(&GlobalInputTokens, uint64(inputTokens))
	atomic.AddUint64(&GlobalOutputTokens, uint64(outputTokens))

	// Log the running total to the console
	log.Printf("[Token Stats] Request: %d / %d | Global Total: %d / %d",
		inputTokens, outputTokens,
		atomic.LoadUint64(&GlobalInputTokens), atomic.LoadUint64(&GlobalOutputTokens))

	// extract individual tool calls with self-correction
	//maxRetries := 10
	//for _ = range maxRetries {
	//fmt.Printf("Prompt tool result: %+v\n", res.Tools)
	extractedCalls, err := bfcl.GetToolCalls(res, bfclTools)
	//fmt.Printf("Extracted tool calls: %v\n", extractedCalls)
	//}

	// add bellman response to history, EITHER: tool or text
	var toolCallIDs []string
	if err != nil {
		fmt.Printf("!!!!!!!!!!! error occurred: %e", err)
		panic(err)
		//rebuiltHistory = append(rebuiltHistory, prompt.AsAssistant(err.Error()))
	} else if res.IsText() {
		content, err := res.AsText()
		if err != nil {
			log.Fatalf("error: %e", err)
		}
		//fmt.Printf("!!!!!!!!!!! assistant answer: %s\n", content)
		rebuiltHistory = append(rebuiltHistory, prompt.AsAssistant(content))
	} else if res.IsTools() {
		for _, t := range res.Tools {
			//fmt.Printf(" ===== Tool in response: %+v\n\n", t)
			rebuiltHistory = append(rebuiltHistory, prompt.AsToolCall(t.ID, t.Name, t.Argument))
			toolCallIDs = append(toolCallIDs, t.ID)
		}
	}

	//fmt.Printf("\n########## toolman conversation history...\n")
	//for i, m := range rebuiltHistory {
	//	fmt.Printf("msg %v: %+v\n", i, m)
	//}

	resp := BenchmarkResponse{
		ToolCalls:      extractedCalls,
		ToolCallIDs:    toolCallIDs,
		ToolmanHistory: rebuiltHistory,
		Content:        "Tool calls generated", // <-- is this used?
		InputTokens:    res.Metadata.InputTokens,
		OutputTokens:   res.Metadata.OutputTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func PrintRequest(r *http.Request) {
	bodyBytes, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	var prettyJSON bytes.Buffer
	if err := json.Indent(&prettyJSON, bodyBytes, "", "  "); err != nil {
		fmt.Printf("Received Raw Body: %s\n", string(bodyBytes))
	} else {
		fmt.Printf("Received Pretty JSON:\n%s\n", prettyJSON.String())
	}
}
