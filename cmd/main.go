package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/tools"
	"github.com/modfin/bellman/tools/ptc"
)

// Request payload from BFCL Python Handler
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
}

// Response payload to BFCL
type BenchmarkResponse struct {
	CallStrings  []string `json:"call_strings"`  // The actual tool calls: ["func(a=1)"]
	Content      string   `json:"content"`       // Any thought/text
	InputTokens  int      `json:"input_tokens"`  // Added for tracking
	OutputTokens int      `json:"output_tokens"` // Added for tracking
}

type Result struct {
	Text string `json:"text" json-description:"The final natural text answer to the user's request."`
}

func main() {
	// Load Env (simulated from your TestToolman)
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

	//fmt.Printf("request struct: %+v", req)

	// 1. Initialize your Client (Reusing your logic)
	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	client := bellman.New(bellmanUrl, bellman.Key{Name: "bfcl", Token: bellmanToken})

	// 2. Convert BFCL Tools (JSON) -> Your PTC Tools
	// Since you said you have a parser:
	bfclTools := ptc.ParseJsonSchemaTools(req.Tools, req.EnablePTC)

	// 3. Construct the Prompt from History
	// BFCL sends a list of messages. Your agent.Run usually takes a single prompt
	// or you might have a chat history method.
	// Assuming we combine history or take the last user message for the "prompt":
	var messages []prompt.Prompt
	for _, msg := range req.Messages {
		switch msg.Role {
		case "user":
			messages = append(messages, prompt.AsUser(msg.Content))
		case "assistant":
			messages = append(messages, prompt.AsAssistant(msg.Content))
		default:
			fmt.Printf("error: unsupported role: %v", msg.Role)
			//case "tool":
			//	messages = append(messages, prompt.AsToolCall(msg.Content))
			//case "tool_response":
			//	messages = append(messages, prompt.AsToolResponse(msg.Content))
		}
	}
	// Append the specific instruction if it's not the last message
	// (Usually the last message is the active instruction)

	// 4. Configure LLM
	model, err := gen.ToModel(req.Model)
	if err != nil {
		log.Fatalf("error: %e", err)
	}

	llm := client.Generator().Model(model).
		System(req.SystemPrompt). // Use passed system prompt or your default
		SetTools(bfclTools...).
		SetPTCLanguage(tools.JavaScript). // Or whatever your default is
		Temperature(req.Temperature)

	// 5. Run Agent
	// Using your RunWithToolsOnly logic since this is a func-call benchmark
	fmt.Printf("time to prompt... %v", messages)
	res, err := llm.Prompt(messages...)
	//res, err := agent.RunWithToolsOnly[Result](10, 0, llm, prompt.AsUser(userPrompt))
	if err != nil {
		log.Printf("Prompt Error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	//fmt.Printf("Tool call result: %v", res)
	prettyJSON, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling: %v\n", err)
	} else {
		fmt.Printf("Tool call result:\n%s\n", string(prettyJSON))
	}

	// 6. Extract the Tool Call String
	// I am assuming 'res' has a field that contains the generated function string.
	// Modify 'GetToolCallString' to match your actual Result struct.
	toolCallStrings := GetToolCallStrings(res)

	resp := BenchmarkResponse{
		CallStrings:  toolCallStrings,
		Content:      strings.Join(toolCallStrings, "; "),
		InputTokens:  res.Metadata.InputTokens,
		OutputTokens: res.Metadata.OutputTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Helper to extract the string "func(arg=1)" from your Agent Result
func GetToolCallStrings(res *gen.Response) []string {
	var calls []string
	if res.IsTools() {
		for _, tool := range res.Tools {
			var argsMap map[string]interface{}
			err := json.Unmarshal(tool.Argument, &argsMap)
			if err != nil {
				// Fallback if not a JSON object
				calls = append(calls, fmt.Sprintf("%s(%s)", tool.Name, string(tool.Argument)))
				continue
			}

			var args []string
			for k, v := range argsMap {
				// Simple formatting for BFCL
				valBytes, _ := json.Marshal(v)
				valStr := string(valBytes)
				args = append(args, fmt.Sprintf("%s=%s", k, valStr))
			}
			sort.Strings(args)
			calls = append(calls, fmt.Sprintf("%s(%s)", tool.Name, strings.Join(args, ", ")))
		}
	} else {
		text, _ := res.AsText()
		if text != "" {
			calls = append(calls, text)
		}
	}
	return calls
}
