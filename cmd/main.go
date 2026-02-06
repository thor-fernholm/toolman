package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/tools"
	"github.com/modfin/bellman/tools/ptc/bfcl"
)

type BenchmarkRequest struct {
	Model        string        `json:"bellman_model"`
	Messages     []Message     `json:"messages"`
	Tools        []interface{} `json:"tools"` // Raw JSON schema tools
	Temperature  float64       `json:"temperature"`
	SystemPrompt string        `json:"system_prompt"`
	EnablePTC    bool          `json:"enable_ptc"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name"`
	ToolID  string `json:"tool_call_id"`
}

type BenchmarkResponse struct {
	ToolCalls    []bfcl.ExtractedCall `json:"tool_calls"`
	CallStrings  []string             `json:"call_strings"`  // The actual tool calls: ["func(a=1)"]
	Content      string               `json:"content"`       // Any thought/text
	InputTokens  int                  `json:"input_tokens"`  // Added for tracking
	OutputTokens int                  `json:"output_tokens"` // Added for tracking
}

func main() {
	// godotenv.Load() ...
	err := godotenv.Load()
	if err != nil {
		panic(err)
	}

	http.HandleFunc("/generate", handleGenerate)

	fmt.Println("Toolman Benchmark Server running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	//bodyBytes, _ := io.ReadAll(r.Body)
	//r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	//var prettyJSON bytes.Buffer
	//if err := json.Indent(&prettyJSON, bodyBytes, "", "  "); err != nil {
	//	fmt.Printf("Received Raw Body: %s\n", string(bodyBytes))
	//} else {
	//	fmt.Printf("Received Pretty JSON:\n%s\n", prettyJSON.String())
	//}

	var req BenchmarkRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	client := bellman.New(bellmanUrl, bellman.Key{Name: "bfcl", Token: bellmanToken})

	//fmt.Println("Request tools: %v", req.Tools)
	bfclTools := bfcl.ParseJsonSchemaTools(req.Tools, req.EnablePTC)

	var messages []prompt.Prompt
	for _, msg := range req.Messages {
		switch msg.Role {
		case "user":
			messages = append(messages, prompt.AsUser(msg.Content))
		case "assistant":
			messages = append(messages, prompt.AsAssistant(msg.Content))
		case "tool":
			//fmt.Printf("Tool call message role: %v, content: %v\n", msg.Role, msg)
			//messages = append(messages, prompt.AsToolCall(msg.ToolID, msg.Name, []byte(msg.Content)))
		//case "tool_response":
		default:
			fmt.Printf("error: unsupported role: %v\n", msg.Role)
			//case "tool":
			//	messages = append(messages, prompt.AsToolCall(msg.Content))
			//case "tool_response":
			//	messages = append(messages, prompt.AsToolResponse(msg.Content))
		}
	}

	model, err := gen.ToModel(req.Model)
	if err != nil {
		log.Fatalf("error: %e", err)
	}

	llm := client.Generator().Model(model).
		System(req.SystemPrompt). // Use passed system prompt or default
		SetTools(bfclTools...).
		SetPTCLanguage(tools.JavaScript).
		Temperature(req.Temperature)

	res, err := llm.Prompt(messages...)
	if err != nil {
		log.Printf("Prompt Error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	extractedCalls := bfcl.GetToolCalls(res, bfclTools)

	var callStrings []string
	for _, call := range extractedCalls {
		for funcName, args := range call {
			callStrings = append(callStrings, bfcl.FormatAsPythonCall(funcName, args))
		}
	}

	resp := BenchmarkResponse{
		ToolCalls:    extractedCalls,
		CallStrings:  callStrings,
		Content:      "Tool calls generated",
		InputTokens:  res.Metadata.InputTokens,
		OutputTokens: res.Metadata.OutputTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
