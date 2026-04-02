package test

import (
	"context"
	"fmt"
	"log"
	"os"
	"testing"
	"time"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman/services/openai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

func TestOpenTelemetryFixedResponse(t *testing.T) {
	err := godotenv.Load("../../../.env")
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	ctx := context.Background()
	system := "# Role\nYou are a helpful LLM assistant."
	model := openai.GenModel_gpt5_mini_latest

	tp := setupHttpLangfuse(ctx, t)
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := tp.Shutdown(shutdownCtx); err != nil {
			t.Errorf("Error shutting down tracer: %v", err)
		}
	}()

	tracer := otel.Tracer("test-suite")
	ctx, parentSpan := tracer.Start(ctx, "Agent_Session")
	defer parentSpan.End()

	parentSpan.SetAttributes(
		attribute.String("langfuse.session.id", "session-fixed-response"),
		attribute.String("gen_ai.system_instructions", system),
	)

	userPrompt := "Convert 69 USD to SEK, generate a secret password, and summarize the result."
	fixedToolName := "code_execution"
	fixedToolArgs := `{"code":"({ exchangeRate: 10.75, sek: 741.75, password: \"otter-sky-42\" })"}`
	fixedToolResult := `{"exchangeRate":10.75,"sek":741.75,"password":"otter-sky-42"}`
	fixedLLMResponse := "69 USD is approximately 741.75 SEK. A generated password is otter-sky-42."

	start := time.Now()
	cursorTime := start

	_, llmSpan := tracer.Start(ctx, fmt.Sprintf("chat %s", model.Name), trace.WithTimestamp(cursorTime))
	llmSpan.SetAttributes(
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.provider.name", model.Provider),
		attribute.String("gen_ai.request.model", model.Name),
		attribute.String("gen_ai.prompt", userPrompt),
	)
	cursorTime = cursorTime.Add(100 * time.Millisecond)

	llmSpan.SetAttributes(
		attribute.String("gen_ai.completion", fmt.Sprintf("Tool Call Requested: %s", fixedToolName)),
	)
	llmSpan.End(trace.WithTimestamp(cursorTime))
	cursorTime = cursorTime.Add(100 * time.Millisecond)

	_, toolSpan := tracer.Start(ctx, fmt.Sprintf("execute_tool %s", fixedToolName), trace.WithTimestamp(cursorTime))
	toolSpan.SetAttributes(
		attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", fixedToolName),
		attribute.String("gen_ai.tool.call.arguments", fixedToolArgs),
		attribute.String("gen_ai.tool.call.id", "call-fixed-1"),
	)
	cursorTime = cursorTime.Add(100 * time.Millisecond)

	toolSpan.SetAttributes(
		attribute.String("gen_ai.tool.call.result", fixedToolResult),
	)
	toolSpan.End(trace.WithTimestamp(cursorTime))
	cursorTime = cursorTime.Add(100 * time.Millisecond)

	_, finalLLMSpan := tracer.Start(ctx, fmt.Sprintf("chat %s", model.Name), trace.WithTimestamp(cursorTime))
	finalLLMSpan.SetAttributes(
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.provider.name", model.Provider),
		attribute.String("gen_ai.request.model", model.Name),
		attribute.String("gen_ai.prompt", fmt.Sprintf("Tool Result: %s", fixedToolResult)),
	)
	cursorTime = cursorTime.Add(100 * time.Millisecond)

	finalLLMSpan.SetAttributes(
		attribute.String("gen_ai.completion", fixedLLMResponse),
		attribute.Int("gen_ai.usage.input_tokens", 64),
		attribute.Int("gen_ai.usage.output_tokens", 22),
		attribute.Int("gen_ai.usage.thinking_tokens", 20),
		attribute.Int64("usage.llm_latency", 44),
	)
	finalLLMSpan.End(trace.WithTimestamp(cursorTime))

	parentSpan.SetAttributes(
		attribute.String("test.name", t.Name()),
		attribute.String("test.langfuse.host", os.Getenv("LANGFUSE_BASE_URL")),
	)
}
