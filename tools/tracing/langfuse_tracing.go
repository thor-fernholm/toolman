package tracing

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"time"

	langfuse "github.com/henomis/langfuse-go"
	"github.com/henomis/langfuse-go/model"
)

type LangfuseTracer struct {
	lf *langfuse.Langfuse
}

func NewLangfuseTracer(ctx context.Context) *LangfuseTracer {
	if os.Getenv("LANGFUSE_HOST") == "" ||
		os.Getenv("LANGFUSE_PUBLIC_KEY") == "" ||
		os.Getenv("LANGFUSE_SECRET_KEY") == "" {
		return &LangfuseTracer{lf: nil}
	}
	return &LangfuseTracer{lf: langfuse.New(ctx)}
}

func (t *LangfuseTracer) Enabled() bool { return t != nil && t.lf != nil }

func (t *LangfuseTracer) Flush(ctx context.Context) {
	if !t.Enabled() {
		return
	}
	t.lf.Flush(ctx)
}

type RunInfo struct {
	TraceID   string
	TraceName string
	RootSpan  string

	Benchmark string
	TaskID    string

	PTCEnabled  bool
	Model       string
	Temperature float64
	MaxTokens   int

	ToolsetSize int
}

type Run struct {
	tracer   *LangfuseTracer
	traceID  string
	rootSpan *model.Span
	start    time.Time
}

// StartRun creates the Trace + root span. If tracing is disabled, returns a no-op Run.
func (t *LangfuseTracer) StartRun(ctx context.Context, info RunInfo, input map[string]any) *Run {
	r := &Run{tracer: t, traceID: info.TraceID, start: time.Now().UTC()}

	if !t.Enabled() {
		return r
	}

	if info.TraceID == "" {
		info.TraceID = fmt.Sprintf("run-%d", time.Now().UnixNano())
		r.traceID = info.TraceID
	}

	// 1) Trace (explicit ID is best)
	_, _ = t.lf.Trace(&model.Trace{
		ID:   info.TraceID,
		Name: nonEmpty(info.TraceName, "toolman.run"),
		Input: model.M{
			"input":     input,
			"benchmark": info.Benchmark,
			"task_id":   info.TaskID,
		},
	})

	// 2) Root span for the whole run
	st := time.Now().UTC()
	root, _ := t.lf.Span(&model.Span{
		TraceID:   info.TraceID,
		Name:      nonEmpty(info.RootSpan, "run"),
		StartTime: &st,
		Metadata: model.M{
			"ptc.enabled":  info.PTCEnabled,
			"model":        info.Model,
			"temperature":  info.Temperature,
			"max_tokens":   info.MaxTokens,
			"toolset.size": info.ToolsetSize,
			"benchmark":    info.Benchmark,
			"task_id":      info.TaskID,
		},
	}, nil)

	r.rootSpan = root
	return r
}

// End closes the root span and attaches a success/error summary + optional output.
func (r *Run) End(err error, output map[string]any) {
	if r == nil || !r.tracer.Enabled() || r.rootSpan == nil {
		return
	}

	end := time.Now().UTC()
	meta := model.M{
		"success":    err == nil,
		"runtime_ms": time.Since(r.start).Milliseconds(),
	}
	if err != nil {
		meta["error.msg"] = err.Error()
		meta["error.type"] = fmt.Sprintf("%T", err)
	}

	_, _ = r.tracer.lf.SpanEnd(&model.Span{
		ID:       r.rootSpan.ID,
		TraceID:  r.traceID,
		EndTime:  &end,
		Metadata: meta,
		Output: model.M{
			"output": output,
		},
	})
}

type LLMCall struct {
	Name         string
	Model        string
	SystemPrompt string
	UserPrompt   string
	Attempt      int
}

type LLMResult struct {
	Text  string
	Usage map[string]any
}

func (r *Run) TraceLLM(ctx context.Context, call LLMCall, fn func(ctx context.Context) (LLMResult, error)) (LLMResult, error) {
	var zero LLMResult
	if r == nil || !r.tracer.Enabled() || r.rootSpan == nil {
		return fn(ctx)
	}

	start := time.Now().UTC()
	gen, _ := r.tracer.lf.Generation(&model.Generation{
		TraceID:   r.traceID,
		Name:      nonEmpty(call.Name, "llm.completion"),
		StartTime: &start,
		Model:     call.Model,
		Input: []model.M{
			{"role": "system", "content": call.SystemPrompt},
			{"role": "user", "content": call.UserPrompt},
		},
		Metadata: model.M{
			"attempt":           call.Attempt,
			"system_prompt_sha": sha(call.SystemPrompt),
			"user_prompt_sha":   sha(call.UserPrompt),
		},
	}, &r.rootSpan.ID)

	res, err := fn(ctx)

	end := time.Now().UTC()
	meta := model.M{
		"success": err == nil,
	}
	if err != nil {
		meta["error.msg"] = err.Error()
		meta["error.type"] = fmt.Sprintf("%T", err)
	}
	for k, v := range res.Usage {
		meta[k] = v
	}

	_, _ = r.tracer.lf.GenerationEnd(&model.Generation{
		ID:       gen.ID,
		TraceID:  r.traceID,
		EndTime:  &end,
		Output:   model.M{"completion": res.Text},
		Metadata: meta,
	})

	if err != nil {
		return zero, err
	}
	return res, nil
}

// TraceTool logs a Span around a tool call.
type ToolCall struct {
	Name     string
	ToolName string
	Attempt  int
	Args     any
}

type ToolResult struct {
	Output      any
	ResultBytes int
}

func (r *Run) TraceTool(ctx context.Context, call ToolCall, fn func(ctx context.Context) (ToolResult, error)) (ToolResult, error) {
	var zero ToolResult
	if r == nil || !r.tracer.Enabled() || r.rootSpan == nil {
		return fn(ctx)
	}

	start := time.Now().UTC()
	sp, _ := r.tracer.lf.Span(&model.Span{
		TraceID:   r.traceID,
		Name:      nonEmpty(call.Name, "tool.call"),
		StartTime: &start,
		Metadata: model.M{
			"tool.name": call.ToolName,
			"attempt":   call.Attempt,
		},
		Input: model.M{
			"args": call.Args,
		},
	}, &r.rootSpan.ID)

	res, err := fn(ctx)

	end := time.Now().UTC()
	meta := model.M{
		"success":      err == nil,
		"result_bytes": res.ResultBytes,
	}
	if err != nil {
		meta["error.msg"] = err.Error()
		meta["error.type"] = fmt.Sprintf("%T", err)
	}

	_, _ = r.tracer.lf.SpanEnd(&model.Span{
		ID:       sp.ID,
		TraceID:  r.traceID,
		EndTime:  &end,
		Metadata: meta,
		Output: model.M{
			"output": res.Output,
		},
	})

	if err != nil {
		return zero, err
	}
	return res, nil
}

type ExecCall struct {
	Name      string // "exec.goja"
	Language  string
	ScriptLen int
	TimeoutMs int
	Attempt   int
}

type ExecResult struct {
	Output any
}

func (r *Run) TraceExec(ctx context.Context, call ExecCall, fn func(ctx context.Context) (ExecResult, error)) (ExecResult, error) {
	var zero ExecResult
	if r == nil || !r.tracer.Enabled() || r.rootSpan == nil {
		return fn(ctx)
	}

	start := time.Now().UTC()
	sp, _ := r.tracer.lf.Span(&model.Span{
		TraceID:   r.traceID,
		Name:      nonEmpty(call.Name, "exec.goja"),
		StartTime: &start,
		Metadata: model.M{
			"exec.language":   call.Language,
			"exec.script_len": call.ScriptLen,
			"exec.timeout_ms": call.TimeoutMs,
			"attempt":         call.Attempt,
			"script_len":      call.ScriptLen,
			"scripy_sha":      call.ScriptLen,
		},
	}, &r.rootSpan.ID)

	res, err := fn(ctx)

	end := time.Now().UTC()
	meta := model.M{
		"success": err == nil,
	}
	if err != nil {
		meta["error.msg"] = err.Error()
		meta["error.type"] = fmt.Sprintf("%T", err)
	}

	_, _ = r.tracer.lf.SpanEnd(&model.Span{
		ID:       sp.ID,
		TraceID:  r.traceID,
		EndTime:  &end,
		Metadata: meta,
		Output: model.M{
			"output": res.Output,
			"status": "Ok",
		},
	})

	if err != nil {
		return zero, err
	}
	return res, nil
}

func (r *Run) Event(name string, meta map[string]any) {
	if r == nil || !r.tracer.Enabled() || r.rootSpan == nil {
		return
	}
	m := model.M{}
	for k, v := range meta {
		m[k] = v
	}
	_, _ = r.tracer.lf.Event(&model.Event{
		Name:     name,
		TraceID:  r.traceID,
		Metadata: m,
	}, &r.rootSpan.ID)
}

func nonEmpty(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

func sha(s string) string {
	if s == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

type runKey struct{}

func WithRun(ctx context.Context, r *Run) context.Context {
	return context.WithValue(ctx, runKey{}, r)
}

func RunFromContext(ctx context.Context) *Run {
	v := ctx.Value(runKey{})
	if r, ok := v.(*Run); ok {
		return r
	}
	return nil
}

func (r *Run) EventIO(name string, input any, output any, meta map[string]any) {
	if r == nil || !r.tracer.Enabled() || r.rootSpan == nil {
		return
	}
	m := model.M{}
	for k, v := range meta {
		m[k] = v
	}

	_, _ = r.tracer.lf.Event(&model.Event{
		Name:     name,
		TraceID:  r.traceID,
		Metadata: m,
		Input: model.M{
			"value": input,
		},
		Output: model.M{
			"value": output,
		},
	}, &r.rootSpan.ID)
}
