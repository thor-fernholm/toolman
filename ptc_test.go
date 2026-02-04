package bellman

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/dop251/goja"
	"github.com/joho/godotenv"
	"github.com/modfin/bellman/agent"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/services/anthropic"
	"github.com/modfin/bellman/services/openai"
	"github.com/modfin/bellman/services/vertexai"
	"github.com/modfin/bellman/tools"
)

func TestToolman(t *testing.T) {
	// get env vars
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")

	allTools := getMockBellmanTools()
	models := []gen.Model{openai.GenModel_gpt4o_mini, vertexai.GenModel_gemini_2_5_flash_latest, anthropic.GenModel_3_haiku_20240307}

	// create Bellman llm and run agent
	client := New(bellmanUrl, Key{Name: "test", Token: bellmanToken})
	llm := client.Generator().System("## Role\nYou are a Financial Assistant. Today is 2026-02-03.").
		SetTools(allTools...).SetPTCLanguage(tools.JavaScript)

	userPrompt := "Predict the future, convert 69 usd to sek, and then generate a secret password."

	// run all models
	for _, m := range models {
		// swap model
		llm = llm.Model(m)

		var res *agent.Result[Result]
		switch llm.Request.Model.Provider {
		case vertexai.Provider:
			res, err = agent.RunWithToolsOnly[Result](5, 0, llm, prompt.AsUser(userPrompt))
		default:
			res, err = agent.Run[Result](5, 0, llm, prompt.AsUser(userPrompt))
		}

		if err != nil {
			log.Fatalf("Prompt() error = %v", err)
		}

		// pretty print
		prettyPrint(res)
	}
}

func TestAutoPTC(t *testing.T) {
	// get env vars
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")

	// setup goja
	vm := goja.New()
	vm.Set("CONFIG", map[string]string{
		"token": bellmanToken,
		"url":   bellmanUrl,
	})
	vm.Set("goLog", func(msg string) { // enables logging to go env
		fmt.Printf("[JS-LOG]: %s\n", msg)
	})

	// define bellman tools
	mockTools := getMockBellmanTools()
	//regularTools, PTCTools := ptc.AdaptToolsToPTC(vm, mockTools, gen.JavaScript)
	//allTools := append(regularTools, PTCTools...)

	// define system prompt
	const systemPrompt = `## Role
You are a Financial Assistant. Today is 2026-02-03.

## Capabilities
You solve complex logic by writing JavaScript code for the code_execution tool.

## Rules for code_execution
1. CALL LIMIT: You may call code_execution ONLY ONCE per turn. 
2. LOGIC: Perform all calculations and multi-tool logic INSIDE the JS script. Before calling code_execution, plan how to combine all tasks into a single script. You are penalized for making more than one tool call.
3. RETURN: The JS script MUST end with an object containing all final data.
4. FORMAT: Do not use console.log for final data; the last evaluated expression is the return value.
6. SYNTHESIS: Once you have the result from code_execution, you have everything you need. 
7. TERMINATION: Do not call the tool again with the same or modified code.
8. SYNC: Do not use async functions unless specified.

## Example JS Script Input
({
  joke: askBellman(CONFIG.url, CONFIG.token, ""),
  total: Sum(123, 456)
})`

	// create Bellman llm and run agent
	client := New(bellmanUrl, Key{Name: "test", Token: bellmanToken})
	llm := client.Generator().Model(openai.GenModel_gpt4o_mini).
		System(systemPrompt).
		SetTools(mockTools...).Temperature(0)

	const useGemini = true // quick-swap provider (gemini separate agent implementation)
	if useGemini {
		llm = llm.Model(vertexai.GenModel_gemini_2_5_flash_latest)
	}

	// prompt, expected result --> <bad-bellman-joke> and 2222222211
	userPrompt := "Predict the future, convert 69 usd to sek, and then generate a secret password."
	type Result struct {
		Text string `json:"text" json-description:"The final natural text answer to the user's request."`
	}
	var res *agent.Result[Result]
	switch llm.Request.Model.Provider {
	case vertexai.Provider:
		res, err = agent.RunWithToolsOnly[Result](10, 0, llm, prompt.AsUser(userPrompt))
	default:
		res, err = agent.Run[Result](10, 0, llm, prompt.AsUser(userPrompt))
	}

	if err != nil {
		log.Fatalf("Prompt() error = %v", err)
	}

	// pretty print
	fmt.Printf("==== %s ====\n", res.Metadata.Model)
	fmt.Printf("==== Result after %d calls ====\n", res.Depth)
	fmt.Printf("%+v\n", res.Result.Text)
	fmt.Printf("==== Conversation ====\n")

	for _, p := range res.Prompts {
		switch p.Role {
		case prompt.ToolCallRole:
			fmt.Printf("%s: %s\n", p.Role, *p.ToolCall)
		case prompt.ToolResponseRole:
			fmt.Printf("%s: %s\n", p.Role, *p.ToolResponse)
		default:
			fmt.Printf("%s: %s\n", p.Role, p.Text)
		}
	}
}

type Result struct {
	Text string `json:"text" json-description:"The final natural text answer to the user's request."`
}

func prettyPrint(res *agent.Result[Result]) {
	fmt.Printf("\n==== %s ====\n", res.Metadata.Model)
	fmt.Printf("==== Result after %d calls ====\n", res.Depth)
	fmt.Printf("%+v\n", res.Result.Text)
	fmt.Printf("==== Conversation ====\n")

	for _, p := range res.Prompts {
		switch p.Role {
		case prompt.ToolCallRole:
			fmt.Printf("%s: %s\n", p.Role, *p.ToolCall)
		case prompt.ToolResponseRole:
			fmt.Printf("%s: %s\n", p.Role, *p.ToolResponse)
		default:
			fmt.Printf("%s: %s\n", p.Role, p.Text)
		}
	}
}

// getMockBellmanTools returns a slice of 3 ready-to-use Bellman tools
func getMockBellmanTools() []tools.Tool {
	var mockTools []tools.Tool

	// 1. Magic 8-Ball Tool
	type FutureArgs struct {
		Question string `json:"question"`
	}
	predictTool := tools.NewTool("predict_future",
		tools.WithDescription("Returns a mystical answer to a yes/no question."),
		tools.WithArgSchema(FutureArgs{}),
		tools.WithPTC(true),
		tools.WithFunction(func(ctx context.Context, call tools.Call) (string, error) {
			var arg FutureArgs
			if err := json.Unmarshal(call.Argument, &arg); err != nil {
				return "", err
			}

			// Logic
			answers := []string{
				"It is certain.", "Reply hazy, try again.", "Don't count on it.",
				"The stars say yes.", "My sources say no.",
			}
			rand.Seed(time.Now().UnixNano())
			return answers[rand.Intn(len(answers))], nil
		}),
	)
	mockTools = append(mockTools, predictTool)

	// 2. Currency Converter Tool
	type CurrencyArgs struct {
		Amount float64 `json:"amount"`
		From   string  `json:"from"`
		To     string  `json:"to"`
	}
	convertTool := tools.NewTool("convert_currency",
		tools.WithDescription("Converts currency amounts (USD, EUR, SEK, GBP, JPY)."),
		tools.WithArgSchema(CurrencyArgs{}),
		tools.WithPTC(true),
		tools.WithFunction(func(ctx context.Context, call tools.Call) (string, error) {
			var arg CurrencyArgs
			if err := json.Unmarshal(call.Argument, &arg); err != nil {
				return "", err
			}

			// Logic
			rates := map[string]float64{"USD": 1.0, "EUR": 0.92, "SEK": 10.50, "GBP": 0.79, "JPY": 148.0}
			rateFrom, ok1 := rates[strings.ToUpper(arg.From)]
			rateTo, ok2 := rates[strings.ToUpper(arg.To)]

			if !ok1 || !ok2 {
				return fmt.Sprintf("Error: Unknown currency pair %s -> %s", arg.From, arg.To), nil
			}
			result := (arg.Amount / rateFrom) * rateTo
			return fmt.Sprintf("%.2f", result), nil
		}),
	)
	mockTools = append(mockTools, convertTool)

	// 3. Password Generator Tool
	type PasswordArgs struct {
		Length  int  `json:"length"`
		Special bool `json:"special"`
	}
	passTool := tools.NewTool("generate_password",
		tools.WithDescription("Generates a random string. 'special' adds symbols."),
		tools.WithArgSchema(PasswordArgs{}),
		tools.WithPTC(true),
		tools.WithFunction(func(ctx context.Context, call tools.Call) (string, error) {
			var arg PasswordArgs
			if err := json.Unmarshal(call.Argument, &arg); err != nil {
				return "", err
			}

			// Logic
			if arg.Length > 50 {
				return "Error: Password too long!", nil
			}
			chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
			if arg.Special {
				chars += "!@#$%^&*()_+"
			}
			var result strings.Builder
			for i := 0; i < arg.Length; i++ {
				idx := (i * 7) % len(chars) // Mock deterministic random
				result.WriteByte(chars[idx])
			}
			return result.String(), nil
		}),
	)
	mockTools = append(mockTools, passTool)

	return mockTools
}

func TestMockPTC(t *testing.T) {
	// get env vars
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	bellmanUrl := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")

	// setup goja
	vm := goja.New()
	vm.Set("CONFIG", map[string]string{
		"token": bellmanToken,
		"url":   bellmanUrl,
	})
	vm.Set("goLog", func(msg string) { // enables logging to go env
		fmt.Printf("[JS-LOG]: %s\n", msg)
	})

	// define go func "tools" for JS use
	askBellman := func(url, token, userMessage string) string {
		client := New(url, Key{Name: "test", Token: token})
		llm := client.Generator()
		res, _ := llm.Model(openai.GenModel_gpt4o_mini).Temperature(1).
			Prompt(
				//prompt.AsUser(userMessage),
				prompt.AsUser("tell me a (very short) Bellman joke in English (Swedish Bellman joke)!"),
			)
		text, _ := res.AsText()
		return text
	}
	// add the "tool"
	vm.Set("askBellman", askBellman)
	// add another "tool"
	Sum := func(a, b int) int {
		return a + b
	}
	vm.Set("Sum", Sum)

	// define PTC tool, input args and func
	type Args struct {
		Code string `json:"code" json-description:"Executable JavaScript source code ONLY. Should use available JS Functions (Environment). Do not include explanations or reasoning."`
	}

	var codeExecutor tools.Function = func(ctx context.Context, call tools.Call) (string, error) {
		// extract tool-call arguments
		var arg Args
		err := json.Unmarshal(call.Argument, &arg)
		if err != nil {
			return "", err
		}

		//fmt.Println("##### JS exec code:\n", arg.Code)

		// run JS code TODO: time limit for loops?
		res, err := vm.RunString(arg.Code)
		if err != nil {
			return fmt.Sprintf(`{"error": %q}`, err.Error()), fmt.Errorf(`{"error": %q}`, err.Error())
		}

		// marshall res into valid JSON
		jsonBytes, err := json.Marshal(res.Export())
		if err != nil {
			return "", err
		}

		return string(jsonBytes), nil
	}

	codeExecution := tools.NewTool("code_execution",
		tools.WithDescription(
			"MANDATORY: You must write executable JavaScript code. "+
				"Executes JS code. Environment has: Sum(a,b), askBellman(url,token,prompt), and CONFIG object. "+
				"Combine all required tool calls into ONE script and return an object with all results.",
		),
		tools.WithArgSchema(Args{}),
		tools.WithFunction(codeExecutor),
	)

	const systemPrompt = `## Role
You are a Financial Assistant. Today is 2026-02-03.

## Capabilities
You solve complex logic by writing JavaScript code for the code_execution tool.

## Rules for code_execution
1. CALL LIMIT: You may call code_execution ONLY ONCE per turn. 
2. LOGIC: Perform all calculations and multi-tool logic INSIDE the JS script. Before calling code_execution, plan how to combine all tasks into a single script. You are penalized for making more than one tool call.
3. RETURN: The JS script MUST end with an object containing all final data.
4. FORMAT: Do not use console.log for final data; the last evaluated expression is the return value.
6. SYNTHESIS: Once you have the result from code_execution, you have everything you need. 
7. TERMINATION: Do not call the tool again with the same or modified code.
8. SYNC: Do not use async functions unless specified.

## Example JS Script Input
({
  joke: askBellman(CONFIG.url, CONFIG.token, ""),
  total: Sum(123, 456)
})`

	// create Bellman llm and run agent
	client := New(bellmanUrl, Key{Name: "test", Token: bellmanToken})
	llm := client.Generator().Model(openai.GenModel_gpt4o_mini).
		System(systemPrompt).
		SetTools(codeExecution).Temperature(0)

	const useGemini = true // quick-swap provider (gemini separate agent implementation)
	if useGemini {
		llm = llm.Model(vertexai.GenModel_gemini_2_5_flash_latest)
	}

	// prompt, expected result --> <bad-bellman-joke> and 2222222211
	userPrompt := "Tell me a Bellman joke and sum the numbers 1234567890 and 0987654321."
	type Result struct {
		Text string `json:"text" json-description:"The final natural text answer to the user's request."`
	}
	var res *agent.Result[Result]
	switch llm.Request.Model.Provider {
	case vertexai.Provider:
		res, err = agent.RunWithToolsOnly[Result](10, 0, llm, prompt.AsUser(userPrompt))
	default:
		res, err = agent.Run[Result](10, 0, llm, prompt.AsUser(userPrompt))
	}

	if err != nil {
		log.Fatalf("Prompt() error = %v", err)
	}

	// pretty print
	fmt.Printf("==== %s ====\n", res.Metadata.Model)
	fmt.Printf("==== Result after %d calls ====\n", res.Depth)
	fmt.Printf("%+v\n", res.Result.Text)
	fmt.Printf("==== Conversation ====\n")

	for _, p := range res.Prompts {
		switch p.Role {
		case prompt.ToolCallRole:
			fmt.Printf("%s: %s\n", p.Role, *p.ToolCall)
		case prompt.ToolResponseRole:
			fmt.Printf("%s: %s\n", p.Role, *p.ToolResponse)
		default:
			fmt.Printf("%s: %s\n", p.Role, p.Text)
		}
	}
}
