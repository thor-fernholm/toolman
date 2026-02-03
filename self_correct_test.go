package bellman

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	//"math"
	"os"
	"testing"

	"github.com/dop251/goja"
	"github.com/joho/godotenv"
	"github.com/modfin/bellman/agent"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/services/openai"
	"github.com/modfin/bellman/services/vertexai"
	//"github.com/modfin/bellman/services/vllm"
	"github.com/modfin/bellman/tools"
)

func TestMockPTCSelfCorrect(t *testing.T) {
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
		timeLimit := 2 * time.Second

		timer := time.NewTimer(timeLimit)

		defer timer.Stop()

		var done <-chan struct{}
		if ctx != nil {
			done = ctx.Done()
		}

		go func() {
			select {
			case <-timer.C:
				vm.Interrupt("execution timeout")
			case <-done:
				vm.Interrupt("context cancelled")
			}
		}()

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
	/*
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
	*/

	const maxAttempts = 3
	originalPrompt := userPrompt
	var res *agent.Result[Result]
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		runPrompt := originalPrompt
		if attempt > 1 && lastErr != nil {
			runPrompt = fmt.Sprintf("%s\n\n"+
				"The previous attempt failed because the generated JavaScript was not executable.\n"+
				"Error message:\n%s\n\n"+
				"Try again. Fix the JavaScript so it is executable and solves the task.\n"+
				"Return only the tool call / executable JavaScript (no explanations).",
				originalPrompt,
				lastErr.Error(),
			)
		}
		switch llm.Request.Model.Provider {
		case vertexai.Provider:
			res, err = agent.RunWithToolsOnly[Result](10, 0, llm, prompt.AsUser(runPrompt))
		default:
			res, err = agent.Run[Result](10, 0, llm, prompt.AsUser(runPrompt))
		}

		if err == nil {
			break

		}

		lastErr = err
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
