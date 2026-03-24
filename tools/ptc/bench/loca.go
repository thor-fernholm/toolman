package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/modfin/bellman"
	"github.com/modfin/bellman/models/gen"
	"github.com/modfin/bellman/prompt"
	"github.com/modfin/bellman/schema"
	"github.com/modfin/bellman/tools"
	"github.com/modfin/bellman/tools/ptc"
)

// LOCA-bench endpoint: POST /loca

type locaRequest struct {
	BellmanModel string `json:"bellman_model"`
	Query        string `json:"query"`

	// Optional allow-list of tool names. If empty, all discovered MCP tools are available.
	Tools []string `json:"tools,omitempty"`

	Temperature  float64 `json:"temperature"`
	MaxTokens    int     `json:"max_tokens"`
	SystemPrompt string  `json:"system_prompt"`
	EnablePTC    bool    `json:"enable_ptc"`
	ToolChoice   string  `json:"tool_choice,omitempty"`

	MCPServers []string `json:"mcp_servers"`
	TimeoutMS  int      `json:"timeout_ms"`
}

type locaTraceCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type locaMetrics struct {
	LatencyMS    int64 `json:"latency_ms"`
	InputTokens  int   `json:"input_tokens"`
	OutputTokens int   `json:"output_tokens"`
}

type locaResponse struct {
	BFCLTrace []locaTraceCall `json:"bfcl_trace"`
	ToolTrace []locaTraceCall `json:"tool_trace"`
	PTCCode   string          `json:"ptc_code,omitempty"`
	Final     any             `json:"final_answer,omitempty"`
	Error     string          `json:"error"`
	Metrics   locaMetrics     `json:"metrics"`

	// Extra fields used by the existing debug UI middleware.
	ToolCalls      []locaTraceCall `json:"tool_calls,omitempty"`
	ToolmanHistory []prompt.Prompt `json:"toolman_history,omitempty"`
	InputTokens    int             `json:"input_tokens,omitempty"`
	OutputTokens   int             `json:"output_tokens,omitempty"`
}

type locaTraceCollector struct {
	mu    sync.Mutex
	calls []locaTraceCall
}

func (c *locaTraceCollector) Add(name string, args map[string]any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.calls = append(c.calls, locaTraceCall{Name: name, Arguments: args})
	c.mu.Unlock()
}

func (c *locaTraceCollector) Snapshot() []locaTraceCall {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]locaTraceCall, len(c.calls))
	copy(out, c.calls)
	return out
}

type locaEvidencePolicy struct {
	enabled bool

	requireMemory           bool
	requireAnnouncements    bool
	requireSubmissionStatus bool
	requireCourseList       bool
	requireAssignments      bool
	requireQuizzes          bool
	requireAssignmentCSV    bool
	requireQuizCSV          bool

	readMemory                bool
	readAnnouncements         bool
	readSubmissionStatus      bool
	attemptedSubmissionStatus bool
	submissionSoftFailed      bool
	listedCourses             bool
	listedAssignments         bool
	listedQuizzes             bool
	readAssignmentCSV         bool
	readQuizCSV               bool
	wroteAssignmentCSV        bool
	wroteQuizCSV              bool
}

type locaWorkspaceState struct {
	mu        sync.RWMutex
	root      string
	serverURL string
	mcp       *mcpClient
}

type locaCourseEvidence struct {
	ID   string
	Code string
	Name string
}

type locaTaskEvidence struct {
	Kind              string
	CourseID          string
	CourseCode        string
	CourseName        string
	Title             string
	Description       string
	Deadline          string
	PointsPossible    string
	Credits           string
	NumberOfQuestions string
	TimeLimit         string
	AllowedAttempts   string
	ScoringPolicy     string
}

type locaSemanticState struct {
	mu          sync.RWMutex
	coursesByID map[string]locaCourseEvidence
	assignments []locaTaskEvidence
	quizzes     []locaTaskEvidence
}

func newLOCASemanticState() *locaSemanticState {
	return &locaSemanticState{
		coursesByID: map[string]locaCourseEvidence{},
	}
}

func newLOCAWorkspaceState(mcp *mcpClient, reg *toolRegistry, initialRoot string) *locaWorkspaceState {
	state := &locaWorkspaceState{
		root: normalizeAllowedRootLOCA(initialRoot),
		mcp:  mcp,
	}
	if serverURL, ok := findMCPServerForOrigToolLOCA(reg, "list_allowed_directories"); ok {
		state.serverURL = serverURL
	}
	return state
}

func (s *locaWorkspaceState) Root() string {
	if s == nil {
		return ""
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.root
}

func (s *locaWorkspaceState) Refresh(ctx context.Context) (string, error) {
	if s == nil {
		return "", fmt.Errorf("workspace state unavailable")
	}
	if s.mcp == nil || strings.TrimSpace(s.serverURL) == "" {
		return s.Root(), fmt.Errorf("list_allowed_directories unavailable")
	}
	out, err := s.mcp.CallTool(ctx, s.serverURL, "list_allowed_directories", map[string]any{})
	if err != nil {
		return "", err
	}
	root := extractAllowedWorkspaceRootLOCA(out)
	if root == "" {
		return "", fmt.Errorf("no allowed directory returned")
	}
	s.mu.Lock()
	s.root = root
	s.mu.Unlock()
	return root, nil
}

func (s *locaSemanticState) ObserveToolResult(name string, args map[string]any, out any) {
	if s == nil {
		return
	}
	toolName := strings.ToLower(strings.TrimSpace(name))
	objects := locaCollectObjectsLOCA(out)
	if len(objects) == 0 {
		return
	}
	if strings.Contains(toolName, "course") {
		for _, obj := range objects {
			s.observeCourse(obj)
		}
	}
	if strings.Contains(toolName, "assignment") {
		for _, obj := range objects {
			s.observeTask("assignment", obj, args)
		}
	}
	if strings.Contains(toolName, "quiz") {
		for _, obj := range objects {
			s.observeTask("quiz", obj, args)
		}
	}
}

func (s *locaSemanticState) HasEvidenceForKind(kind string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	switch kind {
	case "assignment":
		return len(s.assignments) > 0
	case "quiz":
		return len(s.quizzes) > 0
	default:
		return false
	}
}

func (s *locaSemanticState) observeCourse(obj map[string]any) {
	norm := locaNormalizeObjectKeysLOCA(obj)
	course := locaCourseEvidence{
		ID:   locaFirstScalarLOCA(norm["id"], norm["course_id"]),
		Code: locaFirstScalarLOCA(norm["course_code"], norm["code"], norm["course"]),
		Name: locaFirstScalarLOCA(norm["name"], norm["course_name"], norm["display_name"]),
	}
	if course.ID == "" && course.Code == "" && course.Name == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if course.ID != "" {
		prev := s.coursesByID[course.ID]
		if course.Code == "" {
			course.Code = prev.Code
		}
		if course.Name == "" {
			course.Name = prev.Name
		}
		s.coursesByID[course.ID] = course
	}
}

func (s *locaSemanticState) observeTask(kind string, obj map[string]any, args map[string]any) {
	norm := locaNormalizeObjectKeysLOCA(obj)
	rec := locaTaskEvidence{
		Kind:              kind,
		CourseID:          locaFirstScalarLOCA(norm["course_id"], args["course_id"]),
		CourseCode:        locaFirstScalarLOCA(norm["course_code"], args["course_code"]),
		CourseName:        locaFirstScalarLOCA(norm["course_name"]),
		Title:             locaFirstScalarLOCA(norm["assignment_title"], norm["quiz_title"], norm["title"], norm["name"]),
		Description:       locaFirstScalarLOCA(norm["description"], norm["details"], norm["body"], norm["instructions"]),
		Deadline:          locaFirstScalarLOCA(norm["deadline"], norm["due_at"], norm["due_date"]),
		PointsPossible:    locaFirstScalarLOCA(norm["points_possible"], norm["points"], norm["max_points"]),
		Credits:           locaFirstScalarLOCA(norm["credits"], norm["credit"]),
		NumberOfQuestions: locaFirstScalarLOCA(norm["number_of_questions"], norm["question_count"]),
		TimeLimit:         locaFirstScalarLOCA(norm["time_limit"], norm["time_limit_minutes"], norm["duration"]),
		AllowedAttempts:   locaFirstScalarLOCA(norm["allowed_attempts"], norm["attempts_allowed"]),
		ScoringPolicy:     locaFirstScalarLOCA(norm["scoring_policy"], norm["score_policy"], norm["grading_type"]),
	}
	if rec.Title == "" && rec.Description == "" && rec.Deadline == "" {
		return
	}
	if rec.CourseID != "" {
		s.mu.RLock()
		course := s.coursesByID[rec.CourseID]
		s.mu.RUnlock()
		if rec.CourseCode == "" {
			rec.CourseCode = course.Code
		}
		if rec.CourseName == "" {
			rec.CourseName = course.Name
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	switch kind {
	case "assignment":
		s.assignments = locaUpsertTaskEvidenceLOCA(s.assignments, rec)
	case "quiz":
		s.quizzes = locaUpsertTaskEvidenceLOCA(s.quizzes, rec)
	}
}

func locaUpsertTaskEvidenceLOCA(list []locaTaskEvidence, rec locaTaskEvidence) []locaTaskEvidence {
	key := locaTaskEvidenceKeyLOCA(rec)
	for i := range list {
		if locaTaskEvidenceKeyLOCA(list[i]) != key {
			continue
		}
		list[i] = locaMergeTaskEvidenceLOCA(list[i], rec)
		return list
	}
	return append(list, rec)
}

func locaTaskEvidenceKeyLOCA(rec locaTaskEvidence) string {
	return strings.Join([]string{
		rec.Kind,
		strings.ToLower(strings.TrimSpace(rec.CourseID)),
		strings.ToLower(strings.TrimSpace(rec.CourseCode)),
		strings.ToLower(strings.TrimSpace(rec.Title)),
		strings.ToLower(strings.TrimSpace(rec.Deadline)),
	}, "|")
}

func locaMergeTaskEvidenceLOCA(prev, next locaTaskEvidence) locaTaskEvidence {
	if next.CourseID != "" {
		prev.CourseID = next.CourseID
	}
	if next.CourseCode != "" {
		prev.CourseCode = next.CourseCode
	}
	if next.CourseName != "" {
		prev.CourseName = next.CourseName
	}
	if next.Title != "" {
		prev.Title = next.Title
	}
	if next.Description != "" {
		prev.Description = next.Description
	}
	if next.Deadline != "" {
		prev.Deadline = next.Deadline
	}
	if next.PointsPossible != "" {
		prev.PointsPossible = next.PointsPossible
	}
	if next.Credits != "" {
		prev.Credits = next.Credits
	}
	if next.NumberOfQuestions != "" {
		prev.NumberOfQuestions = next.NumberOfQuestions
	}
	if next.TimeLimit != "" {
		prev.TimeLimit = next.TimeLimit
	}
	if next.AllowedAttempts != "" {
		prev.AllowedAttempts = next.AllowedAttempts
	}
	if next.ScoringPolicy != "" {
		prev.ScoringPolicy = next.ScoringPolicy
	}
	return prev
}

func newLOCAEvidencePolicy(query string, reg *toolRegistry) *locaEvidencePolicy {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return &locaEvidencePolicy{}
	}

	enabled := strings.Contains(q, "assignment") ||
		strings.Contains(q, "quiz") ||
		strings.Contains(q, "canvas") ||
		strings.Contains(q, ".csv") ||
		strings.Contains(q, "csv file")
	if !enabled {
		return &locaEvidencePolicy{}
	}

	hasMemoryTool := locaRegistryHasToolLike(reg, "memory")
	hasAnnouncementTool := locaRegistryHasToolLike(reg, "announcement")
	hasSubmissionTool := locaRegistryHasToolLike(reg, "submission") || locaRegistryHasToolLike(reg, "submitted") || locaRegistryHasToolLike(reg, "status")
	hasCourseTool := locaRegistryHasToolLike(reg, "canvas_list_courses")
	hasAssignmentTool := locaRegistryHasToolLike(reg, "canvas_list_assignments")
	hasQuizTool := locaRegistryHasToolLike(reg, "canvas_list_quizzes")

	return &locaEvidencePolicy{
		enabled: enabled,

		requireMemory:           strings.Contains(q, "memory") && hasMemoryTool,
		requireAnnouncements:    strings.Contains(q, "announcement") && hasAnnouncementTool,
		requireSubmissionStatus: locaNeedsSubmissionStatusLOCA(q) && hasSubmissionTool,
		requireCourseList:       hasCourseTool && (hasAssignmentTool || hasQuizTool || hasAnnouncementTool),
		requireAssignments:      strings.Contains(q, "assignment") && hasAssignmentTool,
		requireQuizzes:          strings.Contains(q, "quiz") && hasQuizTool,
		requireAssignmentCSV:    strings.Contains(q, "assignment_info.csv") || strings.Contains(q, "assignment"),
		requireQuizCSV:          strings.Contains(q, "quiz_info.csv") || strings.Contains(q, "quiz"),
	}
}

func locaNeedsSubmissionStatusLOCA(q string) bool {
	return strings.Contains(q, "unfinished") ||
		strings.Contains(q, "submission status") ||
		strings.Contains(q, "submitted") ||
		strings.Contains(q, "must submit") ||
		strings.Contains(q, "have to be completed") ||
		strings.Contains(q, "to be completed") ||
		strings.Contains(q, "status")
}

func locaRegistryHasToolLike(reg *toolRegistry, needle string) bool {
	if reg == nil {
		return false
	}
	n := strings.ToLower(strings.TrimSpace(needle))
	if n == "" {
		return false
	}
	for _, orig := range reg.ToolNameToOrig {
		if strings.Contains(strings.ToLower(orig), n) {
			return true
		}
	}
	return false
}

func (p *locaEvidencePolicy) ObserveTool(name string, args map[string]any) {
	if p == nil || !p.enabled {
		return
	}
	n := strings.ToLower(strings.TrimSpace(name))
	target := locaTargetCSVFromArgs(args)

	if strings.Contains(n, "memory") {
		p.readMemory = true
	}
	if strings.Contains(n, "announcement") {
		p.readAnnouncements = true
	}
	if strings.Contains(n, "submission") || strings.Contains(n, "submitted") || strings.Contains(n, "status") {
		p.readSubmissionStatus = true
	}
	if strings.Contains(n, "canvas_list_courses") {
		p.listedCourses = true
	}
	if strings.Contains(n, "canvas_list_assignments") {
		p.listedAssignments = true
	}
	if strings.Contains(n, "canvas_list_quizzes") {
		p.listedQuizzes = true
	}
	if locaLooksLikeReadToolLOCA(n) {
		switch target {
		case "assignment_info.csv":
			p.readAssignmentCSV = true
		case "quiz_info.csv":
			p.readQuizCSV = true
		}
	}
	if locaLooksLikeWriteToolLOCA(n) {
		switch target {
		case "assignment_info.csv":
			p.wroteAssignmentCSV = true
		case "quiz_info.csv":
			p.wroteQuizCSV = true
		}
	}
}

func (p *locaEvidencePolicy) ObserveToolAttempt(name string) {
	if p == nil || !p.enabled {
		return
	}
	n := strings.ToLower(strings.TrimSpace(name))
	if strings.Contains(n, "submission") || strings.Contains(n, "submitted") || strings.Contains(n, "status") {
		p.attemptedSubmissionStatus = true
	}
}

func (p *locaEvidencePolicy) ObserveToolResult(name string, out any, callErr error) {
	if p == nil || !p.enabled {
		return
	}
	n := strings.ToLower(strings.TrimSpace(name))
	if !(strings.Contains(n, "submission") || strings.Contains(n, "submitted") || strings.Contains(n, "status")) {
		return
	}
	if callErr != nil {
		p.submissionSoftFailed = true
		return
	}
	if locaToolOutputLooksMissingLOCA(out) || locaToolOutputLooksValidationErrorLOCA(out) {
		p.submissionSoftFailed = true
	}
}

func locaLooksLikeReadToolLOCA(name string) bool {
	return strings.Contains(name, "read") || strings.Contains(name, "get") || strings.Contains(name, "list")
}

func locaLooksLikeWriteToolLOCA(name string) bool {
	return strings.Contains(name, "write") ||
		strings.Contains(name, "edit") ||
		strings.Contains(name, "update") ||
		strings.Contains(name, "replace") ||
		strings.Contains(name, "append")
}

func locaTargetCSVFromArgs(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	for _, key := range []string{"path", "file_path", "filepath", "filePath", "filename", "file"} {
		if raw, ok := args[key].(string); ok {
			base := strings.ToLower(strings.TrimSpace(raw))
			if i := strings.LastIndexAny(base, "\\/"); i >= 0 {
				base = base[i+1:]
			}
			if base == "assignment_info.csv" || base == "quiz_info.csv" {
				return base
			}
		}
	}
	return ""
}

func (p *locaEvidencePolicy) CheckBeforeTool(name string, args map[string]any) error {
	if p == nil || !p.enabled {
		return nil
	}
	if !locaLooksLikeWriteToolLOCA(strings.ToLower(strings.TrimSpace(name))) {
		return nil
	}
	target := locaTargetCSVFromArgs(args)
	if target == "" {
		return nil
	}
	if missing := p.missingEvidence(); len(missing) > 0 {
		return fmt.Errorf("blocked premature write to %s: gather required evidence first (%s)", target, strings.Join(missing, ", "))
	}
	return nil
}

func (p *locaEvidencePolicy) CheckBeforeFinal() error {
	if p == nil || !p.enabled {
		return nil
	}
	if missing := p.missingEvidence(); len(missing) > 0 {
		return fmt.Errorf("cannot finish yet: required evidence is still missing (%s)", strings.Join(missing, ", "))
	}
	if p.requireAssignmentCSV && !p.wroteAssignmentCSV {
		return fmt.Errorf("cannot finish yet: assignment_info.csv has not been written")
	}
	if p.requireQuizCSV && !p.wroteQuizCSV {
		return fmt.Errorf("cannot finish yet: quiz_info.csv has not been written")
	}
	return nil
}

func (p *locaEvidencePolicy) missingEvidence() []string {
	if p == nil || !p.enabled {
		return nil
	}
	missing := make([]string, 0, 8)
	if p.requireMemory && !p.readMemory {
		missing = append(missing, "memory")
	}
	if p.requireAnnouncements && !p.readAnnouncements {
		missing = append(missing, "announcements")
	}
	if p.requireSubmissionStatus && !p.readSubmissionStatus && !p.attemptedSubmissionStatus {
		missing = append(missing, "submission_status")
	}
	if p.requireCourseList && !p.listedCourses {
		missing = append(missing, "canvas_list_courses")
	}
	if p.requireAssignments && !p.listedAssignments {
		missing = append(missing, "canvas_list_assignments")
	}
	if p.requireQuizzes && !p.listedQuizzes {
		missing = append(missing, "canvas_list_quizzes")
	}
	if p.requireAssignmentCSV && !p.readAssignmentCSV {
		missing = append(missing, "read_assignment_info.csv")
	}
	if p.requireQuizCSV && !p.readQuizCSV {
		missing = append(missing, "read_quiz_info.csv")
	}
	return missing
}

func locaToolOutputLooksValidationErrorLOCA(out any) bool {
	s := strings.ToLower(strings.TrimSpace(locaStringifyToolOutputLOCA(out)))
	if s == "" {
		return false
	}
	return strings.Contains(s, "input validation error") ||
		strings.Contains(s, "not of type") ||
		strings.Contains(s, "validation error") ||
		strings.Contains(s, "\"error\"")
}

func locaToolOutputLooksMissingLOCA(out any) bool {
	s := strings.ToLower(strings.TrimSpace(locaStringifyToolOutputLOCA(out)))
	if s == "" {
		return false
	}
	return strings.Contains(s, "no submission found") ||
		strings.Contains(s, "not found")
}

func locaStringifyToolOutputLOCA(out any) string {
	if out == nil {
		return ""
	}
	switch v := out.(type) {
	case string:
		return v
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func locaEvidenceReminderLOCA(err error) string {
	if err == nil {
		return ""
	}
	return "Continue working. " + err.Error() + ". Read the missing sources, then overwrite both CSV files in sorted deadline order, and only then provide the final answer."
}

func locaCollectObjectsLOCA(v any) []map[string]any {
	out := []map[string]any{}
	seen := map[string]struct{}{}
	var walk func(any)
	walk = func(cur any) {
		switch vv := cur.(type) {
		case map[string]any:
			b, _ := json.Marshal(vv)
			key := string(b)
			if _, ok := seen[key]; !ok {
				seen[key] = struct{}{}
				out = append(out, vv)
			}
			for _, val := range vv {
				walk(val)
			}
		case []any:
			for _, val := range vv {
				walk(val)
			}
		}
	}
	walk(v)
	return out
}

func locaNormalizeObjectKeysLOCA(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[locaNormalizeKeyLOCA(k)] = v
	}
	return out
}

func locaNormalizeKeyLOCA(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "-", "_")
	return s
}

func locaFirstScalarLOCA(vals ...any) string {
	for _, val := range vals {
		if s := locaScalarStringLOCA(val); s != "" {
			return s
		}
	}
	return ""
}

func locaScalarStringLOCA(v any) string {
	switch vv := v.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(vv)
	case json.Number:
		return vv.String()
	case float64:
		return strconv.FormatFloat(vv, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(vv), 'f', -1, 32)
	case int:
		return strconv.Itoa(vv)
	case int64:
		return strconv.FormatInt(vv, 10)
	case int32:
		return strconv.FormatInt(int64(vv), 10)
	case uint64:
		return strconv.FormatUint(vv, 10)
	case uint32:
		return strconv.FormatUint(uint64(vv), 10)
	case bool:
		if vv {
			return "true"
		}
		return "false"
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func locaExtractCSVContentLOCA(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	for _, key := range []string{"content", "contents", "text", "new_text", "replacement", "data"} {
		if s := locaFirstScalarLOCA(args[key]); s != "" {
			return s
		}
	}
	return ""
}

func locaValidateCSVWriteLOCA(args map[string]any, semantic *locaSemanticState) error {
	if semantic == nil {
		return nil
	}
	target := locaTargetCSVFromArgs(args)
	if target == "" {
		return nil
	}
	content := locaExtractCSVContentLOCA(args)
	if strings.TrimSpace(content) == "" {
		return nil
	}
	reader := csv.NewReader(strings.NewReader(content))
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil || len(rows) == 0 {
		return nil
	}
	headers := rows[0]
	if len(headers) == 0 {
		return nil
	}
	kind := locaInferCSVKindLOCA(target, headers)
	if kind == "" {
		return nil
	}
	if !locaCSVHasMeaningfulDataRowsLOCA(rows[1:]) {
		if semantic.HasEvidenceForKind(kind) {
			return fmt.Errorf("csv write blocked: %s.csv cannot be header-only or empty after evidence for %s rows has been fetched", strings.TrimSuffix(target, ".csv"), kind)
		}
		return nil
	}
	findings := semantic.ValidateRows(kind, headers, rows[1:])
	if len(findings) == 0 {
		return nil
	}
	return fmt.Errorf("csv write blocked: %s", strings.Join(findings, "; "))
}

func locaCSVHasMeaningfulDataRowsLOCA(rows [][]string) bool {
	for _, row := range rows {
		for _, cell := range row {
			if strings.TrimSpace(cell) != "" {
				return true
			}
		}
	}
	return false
}

func locaInferCSVKindLOCA(target string, headers []string) string {
	normHeaders := make([]string, 0, len(headers))
	for _, h := range headers {
		normHeaders = append(normHeaders, locaNormalizeKeyLOCA(h))
	}
	joined := strings.Join(normHeaders, ",")
	switch {
	case strings.Contains(strings.ToLower(target), "assignment") || strings.Contains(joined, "assignment_title") || strings.Contains(joined, "assignment_name"):
		return "assignment"
	case strings.Contains(strings.ToLower(target), "quiz") || strings.Contains(joined, "quiz_title") || strings.Contains(joined, "quiz_name") || strings.Contains(joined, "number_of_questions") || strings.Contains(joined, "time_limit"):
		return "quiz"
	default:
		return ""
	}
}

func (s *locaSemanticState) ValidateRows(kind string, headers []string, rows [][]string) []string {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	var records []locaTaskEvidence
	switch kind {
	case "assignment":
		records = append(records, s.assignments...)
	case "quiz":
		records = append(records, s.quizzes...)
	}
	s.mu.RUnlock()
	if len(records) == 0 {
		return nil
	}
	idx := make(map[string]int, len(headers))
	for i, h := range headers {
		idx[locaNormalizeKeyLOCA(h)] = i
	}
	findings := []string{}
	for rowNo, row := range rows {
		rowMap := locaRowMapLOCA(headers, row)
		match, ok := locaMatchEvidenceForRowLOCA(kind, rowMap, records)
		if !ok {
			findings = append(findings, locaGenericRowFindingLOCA(kind, rowMap, rowNo+2)...)
			continue
		}
		findings = append(findings, locaSemanticFindingsForRowLOCA(match, rowMap, rowNo+2)...)
		if len(findings) >= 4 {
			break
		}
		_ = idx
	}
	return findings
}

func locaRowMapLOCA(headers, row []string) map[string]string {
	out := map[string]string{}
	for i, h := range headers {
		if i >= len(row) {
			out[locaNormalizeKeyLOCA(h)] = ""
			continue
		}
		out[locaNormalizeKeyLOCA(h)] = strings.TrimSpace(row[i])
	}
	return out
}

func locaMatchEvidenceForRowLOCA(kind string, row map[string]string, records []locaTaskEvidence) (locaTaskEvidence, bool) {
	title := locaFirstNonEmptyStringLOCA(row["assignment_title"], row["assignment_name"], row["quiz_title"], row["quiz_name"], row["title"], row["name"])
	courseCode := row["course_code"]
	deadline := row["deadline"]
	bestIdx := -1
	bestScore := -1
	for i, rec := range records {
		if rec.Kind != kind {
			continue
		}
		score := 0
		if title != "" && locaLooseEqualLOCA(title, rec.Title) {
			score += 4
		}
		if courseCode != "" && locaLooseEqualLOCA(courseCode, rec.CourseCode) {
			score += 2
		}
		if deadline != "" && rec.Deadline != "" && locaLooseEqualLOCA(deadline, rec.Deadline) {
			score++
		}
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	if bestIdx >= 0 && bestScore > 0 {
		return records[bestIdx], true
	}
	return locaTaskEvidence{}, false
}

func locaSemanticFindingsForRowLOCA(rec locaTaskEvidence, row map[string]string, rowNo int) []string {
	findings := []string{}
	checkExact := func(header string, expected string, label string) {
		actual, ok := row[header]
		if !ok {
			return
		}
		if expected == "" {
			return
		}
		if strings.TrimSpace(actual) == "" {
			findings = append(findings, fmt.Sprintf("row %d leaves %s empty even though evidence contains %q", rowNo, header, expected))
			return
		}
		if !locaLooseEqualLOCA(actual, expected) {
			findings = append(findings, fmt.Sprintf("row %d maps %s to %q but evidence indicates %s should be %q", rowNo, header, actual, label, expected))
		}
	}
	checkPresent := func(header string, expected string) {
		actual, ok := row[header]
		if !ok || expected == "" {
			return
		}
		if strings.TrimSpace(actual) == "" {
			findings = append(findings, fmt.Sprintf("row %d leaves %s empty even though evidence contains %q", rowNo, header, expected))
		}
	}
	checkExact("course_code", rec.CourseCode, "course code")
	checkExact("course_name", rec.CourseName, "course name")
	if rec.Kind == "assignment" {
		checkExact("assignment_title", rec.Title, "assignment title")
		checkExact("assignment_name", rec.Title, "assignment title")
	} else if rec.Kind == "quiz" {
		checkExact("quiz_title", rec.Title, "quiz title")
		checkExact("quiz_name", rec.Title, "quiz title")
	}
	checkPresent("description", rec.Description)
	checkPresent("deadline", rec.Deadline)
	checkPresent("points_possible", rec.PointsPossible)
	checkPresent("credits", rec.Credits)
	checkPresent("number_of_questions", rec.NumberOfQuestions)
	checkPresent("time_limit", rec.TimeLimit)
	checkPresent("allowed_attempts", rec.AllowedAttempts)
	checkPresent("scoring_policy", rec.ScoringPolicy)
	return findings
}

func locaGenericRowFindingLOCA(kind string, row map[string]string, rowNo int) []string {
	findings := []string{}
	title := row["assignment_title"]
	if kind == "quiz" {
		title = locaFirstNonEmptyStringLOCA(row["quiz_title"], row["quiz_name"], row["title"])
	} else {
		title = locaFirstNonEmptyStringLOCA(row["assignment_title"], row["assignment_name"], row["title"])
	}
	if courseName := row["course_name"]; courseName != "" && title != "" && locaLooseEqualLOCA(courseName, title) {
		findings = append(findings, fmt.Sprintf("row %d appears to use the item title as course_name", rowNo))
	}
	return findings
}

func locaFirstNonEmptyStringLOCA(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func locaLooseEqualLOCA(a, b string) bool {
	an := locaNormalizeValueLOCA(a)
	bn := locaNormalizeValueLOCA(b)
	if an == "" || bn == "" {
		return false
	}
	return an == bn
}

func locaNormalizeValueLOCA(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "_", "")
	s = strings.ReplaceAll(s, "-", "")
	s = strings.ReplaceAll(s, ":", "")
	return s
}

func normalizePTCNullToolResponseLOCA(toolName string, out string, enabled bool) string {
	if !enabled || toolName != ptc.PTCToolName {
		return out
	}
	if strings.TrimSpace(out) != "null" {
		return out
	}
	b, _ := json.Marshal(map[string]any{
		"error": "code_execution returned null. Your JavaScript must return the final result at top level. Do not wrap the whole script in an extra IIFE like (function(){ ... })(); write top-level statements and end with return {...}.",
	})
	return string(b)
}

// --- Minimal MCP client (JSON-RPC 2.0 over HTTP) ---

type mcpClient struct {
	hc *http.Client
}

type mcpToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type mcpRPCReq struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type mcpRPCResp struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Data    any    `json:"data,omitempty"`
	} `json:"error,omitempty"`
}

func (c *mcpClient) postRPC(ctx context.Context, baseURL string, req mcpRPCReq) (mcpRPCResp, error) {
	// Try base URL as-is, then fallback to baseURL + "/mcp".
	try := []string{strings.TrimRight(baseURL, "/")}
	try = append(try, strings.TrimRight(baseURL, "/")+"/mcp")

	body, err := json.Marshal(req)
	if err != nil {
		return mcpRPCResp{}, err
	}

	var lastErr error
	for _, u := range try {
		hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
		if err != nil {
			lastErr = err
			continue
		}
		hreq.Header.Set("Content-Type", "application/json")
		res, err := c.hc.Do(hreq)
		if err != nil {
			lastErr = err
			continue
		}
		b, _ := io.ReadAll(res.Body)
		_ = res.Body.Close()
		if res.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("mcp status %d: %s", res.StatusCode, string(b))
			continue
		}
		var rr mcpRPCResp
		if err := json.Unmarshal(b, &rr); err != nil {
			// Some servers may return non-RPC JSON; wrap it as result.
			return mcpRPCResp{JSONRPC: "2.0", ID: req.ID, Result: b}, nil
		}
		if rr.Error != nil {
			return rr, fmt.Errorf("mcp error %d: %s", rr.Error.Code, rr.Error.Message)
		}
		return rr, nil
	}
	if lastErr == nil {
		lastErr = errors.New("mcp request failed")
	}
	return mcpRPCResp{}, lastErr
}

func (c *mcpClient) ListTools(ctx context.Context, serverURL string) ([]mcpToolDef, error) {
	rr, err := c.postRPC(ctx, serverURL, mcpRPCReq{JSONRPC: "2.0", ID: 1, Method: "tools/list", Params: map[string]any{}})
	if err != nil {
		return nil, err
	}

	// Expected MCP: { result: { tools: [...] } }
	var v struct {
		Tools []mcpToolDef `json:"tools"`
	}
	if err := json.Unmarshal(rr.Result, &v); err == nil && len(v.Tools) > 0 {
		return v.Tools, nil
	}
	// Fallback: result itself is tools array
	var arr []mcpToolDef
	if err := json.Unmarshal(rr.Result, &arr); err == nil && len(arr) > 0 {
		return arr, nil
	}
	// Fallback: whole payload has tools
	var top struct {
		Tools []mcpToolDef `json:"tools"`
	}
	if err := json.Unmarshal(rr.Result, &top); err == nil && len(top.Tools) > 0 {
		return top.Tools, nil
	}
	return []mcpToolDef{}, nil
}

func (c *mcpClient) CallTool(ctx context.Context, serverURL, toolName string, args map[string]any) (any, error) {
	params := map[string]any{"name": toolName, "arguments": args}
	rr, err := c.postRPC(ctx, serverURL, mcpRPCReq{JSONRPC: "2.0", ID: 2, Method: "tools/call", Params: params})
	if err != nil {
		return nil, err
	}

	// Try to decode known MCP shapes.
	var result any
	if err := json.Unmarshal(rr.Result, &result); err == nil {
		return unwrapAndCompactMCPToolResultLOCA(result), nil
	}
	return string(rr.Result), nil
}

func unwrapAndCompactMCPToolResultLOCA(result any) any {
	// MCP tool servers often return wrappers like:
	// {content:[{type:"text", text:"<json>"}], data:..., is_error:false, structured_content:...}
	// Unwrap to a real JSON value (object/array) to reduce token usage and
	// avoid forcing the model to parse JSON encoded as a string.

	// If it's not a wrapper map, we still compact generically.
	inner := unwrapMCPToolWrapperLOCA(result)
	return compactAnyLOCA(inner)
}

func unwrapMCPToolWrapperLOCA(result any) any {
	m, ok := result.(map[string]any)
	if !ok {
		return result
	}

	// Preserve explicit error wrappers.
	if ie, ok := m["is_error"].(bool); ok && ie {
		// Keep a minimal error payload.
		return map[string]any{
			"is_error": true,
			"content":  m["content"],
			"data":     m["data"],
		}
	}

	// Prefer structured_content if present.
	if sc, ok := m["structured_content"]; ok && sc != nil {
		return sc
	}
	// Prefer data if present (some tools put machine-readable output there).
	if d, ok := m["data"]; ok && d != nil {
		return d
	}
	// Some tools use {"content":"..."}.
	if cs, ok := m["content"].(string); ok {
		if v, ok := parseMaybeJSONLOCA(cs); ok {
			return v
		}
		return cs
	}

	// Fallback: parse JSON embedded in content[0].text
	if content, ok := m["content"].([]any); ok && len(content) > 0 {
		if first, ok := content[0].(map[string]any); ok {
			if txt, ok := first["text"].(string); ok {
				if v, ok := parseMaybeJSONLOCA(txt); ok {
					return v
				}
				return txt
			}
		}
	}
	return result
}

func parseMaybeJSONLOCA(s string) (any, bool) {
	ss := strings.TrimSpace(s)
	if ss == "" {
		return nil, false
	}
	// Only attempt if it looks like JSON.
	if !(strings.HasPrefix(ss, "{") || strings.HasPrefix(ss, "[")) {
		return nil, false
	}
	var v any
	if err := json.Unmarshal([]byte(ss), &v); err != nil {
		return nil, false
	}
	return v, true
}

func compactAnyLOCA(v any) any {
	// Generic, size-based compaction. Goal: reduce token usage while keeping
	// results machine-readable across arbitrary MCP servers.
	const (
		maxDepth    = 4
		maxString   = 4000
		maxArrayLen = 50
		maxMapKeys  = 80
	)
	budget := 1500 // max nodes visited
	return compactAnyInnerLOCA(v, 0, &budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
}

func compactAnyInnerLOCA(v any, depth int, budget *int, maxDepth, maxString, maxArrayLen, maxMapKeys int) any {
	if budget == nil {
		return v
	}
	if *budget <= 0 {
		return "<truncated>"
	}
	*budget--

	if v == nil {
		return nil
	}
	if depth >= maxDepth {
		switch vv := v.(type) {
		case string:
			return compactStringLOCA(vv, maxString)
		case bool, float64, float32, int, int64, int32, uint64, uint32:
			return vv
		default:
			b, _ := json.Marshal(v)
			return compactStringLOCA(string(b), maxString)
		}
	}

	switch vv := v.(type) {
	case string:
		if looksLikeDiffLOCA(vv) {
			return summarizeDiffLOCA(vv)
		}
		return compactStringLOCA(vv, maxString)
	case bool, float64, float32, int, int64, int32, uint64, uint32:
		return vv
	case []any:
		if len(vv) > maxArrayLen {
			vv = vv[:maxArrayLen]
		}
		out := make([]any, 0, len(vv))
		for _, it := range vv {
			out = append(out, compactAnyInnerLOCA(it, depth+1, budget, maxDepth, maxString, maxArrayLen, maxMapKeys))
		}
		return out
	case map[string]any:
		keys := make([]string, 0, len(vv))
		for k := range vv {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if len(keys) > maxMapKeys {
			keys = keys[:maxMapKeys]
		}
		out := make(map[string]any, len(keys))
		for _, k := range keys {
			out[k] = compactAnyInnerLOCA(vv[k], depth+1, budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
		}
		return out
	default:
		// Best-effort via JSON roundtrip into map/slice.
		b, err := json.Marshal(v)
		if err == nil {
			var mm map[string]any
			if err := json.Unmarshal(b, &mm); err == nil && mm != nil {
				return compactAnyInnerLOCA(mm, depth, budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
			}
			var aa []any
			if err := json.Unmarshal(b, &aa); err == nil && aa != nil {
				return compactAnyInnerLOCA(aa, depth, budget, maxDepth, maxString, maxArrayLen, maxMapKeys)
			}
			return compactStringLOCA(string(b), maxString)
		}
		return "<unserializable>"
	}
}

func compactStringLOCA(s string, max int) string {
	ss := strings.TrimSpace(s)
	if len(ss) <= max {
		return ss
	}
	head := ss
	if max > 200 {
		head = ss[:max-200]
	}
	tail := ss
	if len(ss) > 200 {
		tail = ss[len(ss)-200:]
	}
	return head + "\n...<truncated>...\n" + tail
}

func looksLikeDiffLOCA(s string) bool {
	ss := strings.TrimSpace(s)
	if ss == "" {
		return false
	}
	if strings.Contains(ss, "```diff") {
		return true
	}
	if strings.Contains(ss, "\n+++ ") && strings.Contains(ss, "\n--- ") && strings.Contains(ss, "\n@@") {
		return true
	}
	if strings.HasPrefix(ss, "Index: ") {
		return true
	}
	return false
}

func summarizeDiffLOCA(s string) any {
	ss := s
	idx := strings.Index(ss, "Index: ")
	target := ""
	if idx >= 0 {
		line := ss[idx:]
		if nl := strings.IndexAny(line, "\r\n"); nl >= 0 {
			line = line[:nl]
		}
		target = strings.TrimSpace(strings.TrimPrefix(line, "Index: "))
	}
	return map[string]any{
		"ok":     true,
		"kind":   "diff",
		"target": target,
	}
}

// --- Tool bootstrap ---

type toolRegistry struct {
	Tools          []tools.Tool
	ToolNameToURL  map[string]string // Bellman tool name -> server URL
	ToolNameToOrig map[string]string // Bellman tool name -> original MCP tool name
}

func mcpSchemaToBellmanSchema(m map[string]any) (*schema.JSON, error) {
	if m == nil {
		return &schema.JSON{Type: schema.Object, Properties: map[string]*schema.JSON{}}, nil
	}

	// Bellman's schema.JSON is intentionally minimal; many MCP servers return
	// full JSON Schema where fields vary in shape. Pre-normalize the map to
	// avoid JSON unmarshal failures and keep only what Bellman understands.
	//
	// Notably:
	// - "type" may be an array (e.g. ["string","null"]) -> translate to
	//   type:"string" + nullable:true
	// - "additionalProperties" may be boolean true/false -> drop it (best-effort)
	sanitizeMCPJSONSchemaLOCA(m)

	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	var s schema.JSON
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	normalizeSchemaLOCA(&s)
	if s.Type == "" {
		if len(s.Properties) > 0 {
			s.Type = schema.Object
		}
	}
	return &s, nil
}

func sanitizeMCPJSONSchemaLOCA(v any) {
	switch vv := v.(type) {
	case map[string]any:
		// Handle JSON Schema: type can be string or array (nullable via "null").
		if tv, ok := vv["type"]; ok {
			switch t := tv.(type) {
			case []any:
				nullable := false
				chosen := ""
				for _, e := range t {
					es, ok := e.(string)
					if !ok {
						continue
					}
					if es == "null" {
						nullable = true
						continue
					}
					if chosen == "" {
						chosen = es
					}
				}
				if chosen != "" {
					vv["type"] = chosen
				} else {
					delete(vv, "type")
				}
				if nullable {
					if nb, ok := vv["nullable"].(bool); ok {
						vv["nullable"] = nb || true
					} else {
						vv["nullable"] = true
					}
				}
			}
		}

		// Handle JSON Schema: additionalProperties can be bool or schema.
		if ap, ok := vv["additionalProperties"]; ok {
			if _, ok := ap.(bool); ok {
				// Best-effort: Bellman schema expects an object schema, not a boolean.
				delete(vv, "additionalProperties")
			}
		}

		for _, it := range vv {
			sanitizeMCPJSONSchemaLOCA(it)
		}
	case []any:
		for _, it := range vv {
			sanitizeMCPJSONSchemaLOCA(it)
		}
	}
}

func normalizeSchemaLOCA(s *schema.JSON) {
	if s == nil {
		return
	}
	if strings.EqualFold(string(s.Type), "dict") {
		s.Type = schema.Object
	}
	for _, p := range s.Properties {
		normalizeSchemaLOCA(p)
	}
	if s.Items != nil {
		normalizeSchemaLOCA(s.Items)
	}
	if s.AdditionalProperties != nil {
		normalizeSchemaLOCA(s.AdditionalProperties)
	}
	for _, d := range s.Defs {
		normalizeSchemaLOCA(d)
	}
}

func sanitizeToolName(name string) string {
	// Bellman/OpenAI-like function names should avoid special chars.
	s := strings.TrimSpace(name)
	if s == "" {
		return "tool"
	}
	// Replace non [a-zA-Z0-9_] with underscore
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'
		if ok {
			b = append(b, c)
		} else {
			b = append(b, '_')
		}
	}
	out := strings.ToLower(string(b))
	out = strings.Trim(out, "_")
	for strings.Contains(out, "__") {
		out = strings.ReplaceAll(out, "__", "_")
	}
	if out == "" {
		out = "tool"
	}
	if out[0] >= '0' && out[0] <= '9' {
		out = "get_" + out
	}
	return out
}

func buildRegistry(ctx context.Context, mcpServers []string, allowList []string, client *mcpClient) (*toolRegistry, error) {
	allow := map[string]bool{}
	if len(allowList) > 0 {
		for _, n := range allowList {
			allow[strings.TrimSpace(n)] = true
		}
	}

	reg := &toolRegistry{
		Tools:          []tools.Tool{},
		ToolNameToURL:  map[string]string{},
		ToolNameToOrig: map[string]string{},
	}
	seen := map[string]bool{}

	for _, rawURL := range mcpServers {
		u := strings.TrimSpace(rawURL)
		if u == "" {
			continue
		}
		defs, err := client.ListTools(ctx, u)
		if err != nil {
			return nil, fmt.Errorf("list tools from %s: %w", u, err)
		}
		for _, d := range defs {
			origName := strings.TrimSpace(d.Name)
			if origName == "" {
				continue
			}
			if len(allow) > 0 {
				// allow matches original names
				if !allow[origName] && !allow[sanitizeToolName(origName)] {
					continue
				}
			}
			name := sanitizeToolName(origName)
			if seen[name] {
				// Collision: prefer first server. Ignore duplicates.
				continue
			}
			seen[name] = true
			s, err := mcpSchemaToBellmanSchema(d.InputSchema)
			if err != nil {
				return nil, fmt.Errorf("tool %s schema: %w", origName, err)
			}

			t := tools.NewTool(name, tools.WithDescription(d.Description))
			t.ArgumentSchema = s
			reg.Tools = append(reg.Tools, t)
			reg.ToolNameToURL[name] = u
			reg.ToolNameToOrig[name] = origName
		}
	}

	if len(reg.Tools) == 0 {
		return nil, errors.New("no tools discovered from mcp_servers")
	}
	return reg, nil
}

// --- Handler / runners ---

func HandleGenerateLOCA(w http.ResponseWriter, r *http.Request) {
	fmt.Println("HandleGenerateLOCA")
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()
	var req locaRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.BellmanModel) == "" {
		http.Error(w, "bellman_model is required", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if len(req.MCPServers) == 0 {
		http.Error(w, "mcp_servers is required", http.StatusBadRequest)
		return
	}

	// Safety: some upstream providers reject too-large max_tokens.
	// Clamp to a commonly supported completion limit for OpenAI-compatible models.
	if req.MaxTokens > 16384 {
		req.MaxTokens = 16384
	}

	// Provider quirks: some models (e.g. OpenAI GPT-5) only support the default
	// temperature value. If the request omits temperature, it unmarshals as 0.
	// Normalize to a supported value to avoid hard failures.
	normalizeModelParamsLOCA(&req)

	ctx := r.Context()
	if req.TimeoutMS > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(req.TimeoutMS)*time.Millisecond)
		defer cancel()
	}

	bellmanURL := os.Getenv("BELLMAN_URL")
	bellmanToken := os.Getenv("BELLMAN_TOKEN")
	if strings.TrimSpace(bellmanURL) == "" || strings.TrimSpace(bellmanToken) == "" {
		http.Error(w, "BELLMAN_URL and BELLMAN_TOKEN must be set", http.StatusInternalServerError)
		return
	}
	bClient := bellman.New(bellmanURL, bellman.Key{Name: "loca", Token: bellmanToken})

	mcp := &mcpClient{hc: &http.Client{Timeout: 60 * time.Second}}
	reg, err := buildRegistry(ctx, req.MCPServers, req.Tools, mcp)
	if err != nil {
		writeLOCAResponse(w, start, nil, nil, "", nil, 0, 0, err)
		return
	}

	// Help the model avoid sandbox path errors by discovering the allowed
	// workspace root and injecting it into the system prompt.
	allowedRoot := injectAllowedWorkspaceRootLOCA(ctx, mcp, reg, &req)

	if req.EnablePTC {
		trace, ptcCode, final, hist, inTok, outTok, err := runLOCAPTC(ctx, bClient, mcp, reg, req, allowedRoot)
		writeLOCAResponse(w, start, trace, hist, ptcCode, final, inTok, outTok, err)
		return
	}

	trace, final, hist, inTok, outTok, err := runLOCANormal(ctx, bClient, mcp, reg, req, allowedRoot)
	fmt.Println("Input tokens ", inTok)
	fmt.Println("Output tokens ", outTok)
	writeLOCAResponse(w, start, trace, hist, "", final, inTok, outTok, err)
}

func normalizeModelParamsLOCA(req *locaRequest) {
	if req == nil {
		return
	}
	model := strings.ToLower(strings.TrimSpace(req.BellmanModel))
	if model == "" {
		return
	}

	// OpenAI GPT-5 currently supports only the default temperature (1).
	// Error example: "Unsupported value: 'temperature' does not support 0 ..."
	if strings.Contains(model, "gpt-5") {
		if req.Temperature != 1 {
			req.Temperature = 1
		}
	}
}

func injectAllowedWorkspaceRootLOCA(ctx context.Context, mcp *mcpClient, reg *toolRegistry, req *locaRequest) string {
	if mcp == nil || reg == nil || req == nil {
		return ""
	}

	serverURL, ok := findMCPServerForOrigToolLOCA(reg, "list_allowed_directories")
	if !ok {
		// Fall back to instruction-only.
		appendWorkspacePathInstructionLOCA(req, "", reg)
		return ""
	}

	out, err := mcp.CallTool(ctx, serverURL, "list_allowed_directories", map[string]any{})
	if err != nil {
		appendWorkspacePathInstructionLOCA(req, "", reg)
		return ""
	}

	root := extractAllowedWorkspaceRootLOCA(out)
	appendWorkspacePathInstructionLOCA(req, root, reg)
	return root
}

func rewriteCSVPathArgsLOCA(args map[string]any, allowedRoot string) map[string]any {
	// Backwards-compatible shim; keep name but apply a recursive rewrite.
	if args == nil {
		return args
	}
	sanitized, ok := sanitizeToolArgsAnyLOCA(args).(map[string]any)
	if !ok {
		sanitized = args
	}
	out, ok := rewriteCSVPathsAnyLOCA(sanitized, allowedRoot).(map[string]any)
	if !ok {
		return sanitized
	}
	return out
}

func normalizeAllowedRootLOCA(root string) string {
	root = strings.TrimSpace(root)
	if root == "" {
		return ""
	}
	root = strings.Trim(root, "\"")
	root = filepath.Clean(root)
	return strings.TrimRight(root, "\\/")
}

func sanitizeToolArgsAnyLOCA(v any) any {
	switch vv := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(vv))
		for k, val := range vv {
			if val == nil {
				continue
			}
			clean := sanitizeToolArgsAnyLOCA(val)
			if clean == nil {
				continue
			}
			out[k] = clean
		}
		return out
	case []any:
		out := make([]any, 0, len(vv))
		for _, it := range vv {
			clean := sanitizeToolArgsAnyLOCA(it)
			if clean != nil {
				out = append(out, clean)
			}
		}
		return out
	default:
		return v
	}
}

func rewriteCSVPathsAnyLOCA(v any, allowedRoot string) any {
	root := normalizeAllowedRootLOCA(allowedRoot)
	if root == "" {
		return v
	}

	// Only rewrite values that are very likely to be file paths.
	pathKeys := map[string]bool{
		"path":      true,
		"file_path": true,
		"filepath":  true,
		"filePath":  true,
		"filename":  true,
		"file":      true,
	}

	switch vv := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(vv))
		for k, val := range vv {
			// Recurse by default.
			newVal := rewriteCSVPathsAnyLOCA(val, root)
			// Additionally rewrite known path-like keys when the value is a string.
			if pathKeys[k] {
				if s, ok := val.(string); ok {
					if rewritten, ok2 := rewriteCSVPathStringLOCA(s, root); ok2 {
						newVal = rewritten
					}
				}
			}
			out[k] = newVal
		}
		return out
	case []any:
		out := make([]any, 0, len(vv))
		for _, it := range vv {
			out = append(out, rewriteCSVPathsAnyLOCA(it, root))
		}
		return out
	case string:
		// Only rewrite bare strings if they *look* like a file path.
		if rewritten, ok := rewriteCSVPathStringLOCA(vv, root); ok {
			return rewritten
		}
		return vv
	default:
		return v
	}
}

func rewriteCSVPathStringLOCA(s string, root string) (string, bool) {
	p := strings.TrimSpace(s)
	if p == "" {
		return s, false
	}
	base := p
	if i := strings.LastIndexAny(base, "\\/"); i >= 0 {
		base = base[i+1:]
	}
	bn := strings.ToLower(strings.TrimSpace(base))
	if bn != "assignment_info.csv" && bn != "quiz_info.csv" {
		return s, false
	}
	// If it is just a filename or a path, force it into the allowed root.
	// Keep original base casing.
	return filepath.Join(normalizeAllowedRootLOCA(root), base), true
}

func findMCPServerForOrigToolLOCA(reg *toolRegistry, origToolName string) (string, bool) {
	if reg == nil {
		return "", false
	}
	needle := strings.TrimSpace(origToolName)
	if needle == "" {
		return "", false
	}
	for sanitized, orig := range reg.ToolNameToOrig {
		if orig == needle {
			if u, ok := reg.ToolNameToURL[sanitized]; ok && strings.TrimSpace(u) != "" {
				return u, true
			}
		}
	}
	// Also accept match on sanitized name.
	san := sanitizeToolName(needle)
	if u, ok := reg.ToolNameToURL[san]; ok && strings.TrimSpace(u) != "" {
		return u, true
	}
	return "", false
}

func appendWorkspacePathInstructionLOCA(req *locaRequest, workspaceRoot string, reg *toolRegistry) {
	// Keep this short; it is purely to prevent path-related tool failures.
	root := strings.TrimSpace(workspaceRoot)
	instr := "IMPORTANT (workspace/files): You MUST edit the existing CSV files 'assignment_info.csv' and 'quiz_info.csv' in the allowed workspace. Do NOT create new CSV files. Use only absolute file paths inside the allowed workspace directory. Never use '/workspace', 'C:\\workspace', '/', or relative paths like 'assignment_info.csv'."
	if root != "" {
		instr += fmt.Sprintf(" Allowed workspace root: %s. When accessing a file, set path to '%s\\\\<filename>'.", root, root)
		instr += fmt.Sprintf(" The two files you must edit are: '%s\\\\assignment_info.csv' and '%s\\\\quiz_info.csv'.", root, root)
	} else {
		instr += " Call list_allowed_directories first, then prefix all file paths with the returned directory."
	}
	instr += " If a filesystem tool reports access denied, invalid path, path outside allowed directories, or no allowed directory returned, immediately call list_allowed_directories again, rebuild the absolute path from that returned directory, and retry once with the corrected path. Do not continue reasoning as if evidence is missing until the path issue is fixed."

	// CSV correctness: prevent "prepend new rows" + leaving stale template/example lines behind.
	instr += " CRITICAL (CSV): Before editing, you MUST read the current contents of BOTH CSV files. Use the EXACT existing header line (line 1) as the header, keep it as the FIRST line, and do NOT invent/rename/reorder columns. Remove any example/template rows that were already in the files (e.g. lines containing '(Example)'). When writing, OVERWRITE the entire file contents (replace the full old text with the full new text) so the final file contains ONLY: the header line + the required data rows. Do not prepend/append snippets. Perform one edit per file that replaces the entire content."
	instr += " Build rows from a canonical intermediate object before serializing to CSV: first normalize each real-world item into semantic fields such as course_code, course_name, title, description, deadline, points_possible, credits, number_of_questions, time_limit, allowed_attempts, and scoring_policy; then map those semantic fields onto the EXACT target header names. Never map by column position or by vaguely similar nearby keys."
	instr += " Header mapping rules: course_code must come from the course's code/identifier; course_name must come from the course's human-readable name; assignment_title or quiz_title must come from the assignment/quiz title or name; description must use the item's descriptive text when available; points_possible, credits, number_of_questions, time_limit, allowed_attempts, and scoring_policy must be filled whenever the fetched item data contains them."
	instr += " Before writing each CSV, audit every column in the header against the source object you are serializing. If a value exists in the fetched evidence, do not leave that CSV field blank. If the source data does not contain a field, only then leave it empty."

	instr += " IMPORTANT (Canvas IDs): Never call course-specific tools with course_id=0. Always call canvas_list_courses first, then iterate over the returned course IDs when calling canvas_list_assignments/canvas_list_quizzes/canvas_list_announcements."
	instr += " IMPORTANT (evidence-first workflow): Do not write any CSV and do not give a final answer until you have completed the required evidence checklist for this task. First read BOTH CSV files to preserve the exact header. Then gather course IDs. Then gather assignments and quizzes for every course. If the task mentions announcements, unfinished work, submission status, or memory, you MUST explicitly fetch those sources before deciding what belongs in the CSVs."
	if locaRegistryHasToolLike(reg, "memory") {
		instr += " If the prompt says personal information is stored in memory, call the memory tool before deciding requirements."
	}
	if locaRegistryHasToolLike(reg, "announcement") {
		instr += " If the prompt says some work may not need submission according to teacher announcements, announcements are mandatory evidence rather than optional context."
	}
	if locaRegistryHasToolLike(reg, "submission") || locaRegistryHasToolLike(reg, "submitted") || locaRegistryHasToolLike(reg, "status") {
		instr += " If the prompt asks for unfinished or required submissions, you MUST check submission-status related tools before writing."
	}
	instr += " IMPORTANT (sorting): Before writing, ensure rows are sorted by deadline ascending; for identical deadlines, sort by class code in dictionary order."

	s := strings.TrimSpace(req.SystemPrompt)
	if s != "" {
		s += "\n\n"
	}
	req.SystemPrompt = s + instr
}

func extractAllowedWorkspaceRootLOCA(out any) string {
	candidates := collectPathCandidatesLOCA(out)
	if len(candidates) == 0 {
		return ""
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		left := strings.ToLower(candidates[i])
		right := strings.ToLower(candidates[j])
		leftScore := strings.Contains(left, "agent_workspace")
		rightScore := strings.Contains(right, "agent_workspace")
		if leftScore != rightScore {
			return leftScore
		}
		return len(left) < len(right)
	})
	for _, candidate := range candidates {
		if root := normalizeAllowedRootLOCA(candidate); root != "" {
			return root
		}
	}
	return ""
}

func collectPathCandidatesLOCA(v any) []string {
	out := []string{}
	seen := map[string]struct{}{}
	var walk func(any)
	walk = func(cur any) {
		switch vv := cur.(type) {
		case string:
			for _, candidate := range extractPathStringsLOCA(vv) {
				if _, ok := seen[candidate]; ok {
					continue
				}
				seen[candidate] = struct{}{}
				out = append(out, candidate)
			}
		case map[string]any:
			for _, val := range vv {
				walk(val)
			}
		case []any:
			for _, val := range vv {
				walk(val)
			}
		default:
			return
		}
	}
	walk(v)
	return out
}

func extractPathStringsLOCA(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	fields := strings.FieldsFunc(text, func(r rune) bool {
		switch r {
		case '\r', '\n', '\t', ' ', '"', '\'', ',', ';', '[', ']', '{', '}', '(', ')':
			return true
		default:
			return false
		}
	})
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		if looksLikeAbsolutePathLOCA(field) {
			out = append(out, field)
		}
	}
	return out
}

func looksLikeAbsolutePathLOCA(path string) bool {
	if len(path) >= 3 && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) && path[1] == ':' && (path[2] == '\\' || path[2] == '/') {
		return true
	}
	if strings.HasPrefix(path, `\\`) {
		return true
	}
	if strings.HasPrefix(path, "/") && strings.Contains(strings.ToLower(path), "workspace") {
		return true
	}
	return false
}

func locaToolTouchesFilesystemLOCA(toolName string, args map[string]any) bool {
	if locaTargetCSVFromArgs(args) != "" {
		return true
	}
	n := strings.ToLower(strings.TrimSpace(toolName))
	if n == "" {
		return false
	}
	return strings.Contains(n, "file") ||
		strings.Contains(n, "directory") ||
		strings.Contains(n, "read_text") ||
		strings.Contains(n, "write_file") ||
		strings.Contains(n, "list_allowed_directories")
}

func resolveToolArgsLOCA(ctx context.Context, state *locaWorkspaceState, toolName string, args map[string]any) (map[string]any, error) {
	resolved := rewriteCSVPathArgsLOCA(args, "")
	if !locaToolTouchesFilesystemLOCA(toolName, resolved) {
		return resolved, nil
	}
	root := ""
	if state != nil {
		root = state.Root()
	}
	if root == "" {
		if state == nil {
			return resolved, fmt.Errorf("no allowed directory returned")
		}
		var err error
		root, err = state.Refresh(ctx)
		if err != nil {
			return resolved, err
		}
	}
	return rewriteCSVPathArgsLOCA(resolved, root), nil
}

func locaPathResolutionErrorLOCA(err error, out any) error {
	if err != nil {
		if perr := locaPathResolutionErrorFromStringLOCA(err.Error()); perr != nil {
			return perr
		}
	}
	if perr := locaPathResolutionErrorFromStringLOCA(locaStringifyToolOutputLOCA(out)); perr != nil {
		return perr
	}
	return nil
}

func locaPathResolutionErrorFromStringLOCA(text string) error {
	s := strings.ToLower(strings.TrimSpace(text))
	if s == "" {
		return nil
	}
	if strings.Contains(s, "path outside allowed directories") ||
		strings.Contains(s, "access denied - path outside allowed directories") ||
		strings.Contains(s, "access denied") ||
		strings.Contains(s, "invalid path") ||
		strings.Contains(s, "no allowed directory returned") {
		return errors.New(strings.TrimSpace(text))
	}
	return nil
}

func writeLOCAResponse(w http.ResponseWriter, start time.Time, trace []locaTraceCall, hist []prompt.Prompt, ptcCode string, final any, inTok, outTok int, err error) {
	resp := locaResponse{
		BFCLTrace:      trace,
		ToolTrace:      trace,
		PTCCode:        ptcCode,
		Final:          final,
		Error:          "",
		Metrics:        locaMetrics{LatencyMS: time.Since(start).Milliseconds(), InputTokens: inTok, OutputTokens: outTok},
		ToolCalls:      trace,
		ToolmanHistory: hist,
		InputTokens:    inTok,
		OutputTokens:   outTok,
	}
	if err != nil {
		resp.Error = err.Error()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func toolChoiceToConfig(choice string, toolMap map[string]tools.Tool) (*tools.Tool, error) {
	c := strings.TrimSpace(choice)
	if c == "" {
		return nil, nil
	}
	switch strings.ToLower(c) {
	case "auto":
		t := tools.AutoTool
		return &t, nil
	case "required":
		t := tools.RequiredTool
		return &t, nil
	default:
		// specific tool
		if t, ok := toolMap[sanitizeToolName(c)]; ok {
			return &t, nil
		}
		if t, ok := toolMap[c]; ok {
			return &t, nil
		}
		return nil, fmt.Errorf("unknown tool_choice %q", choice)
	}
}

type locaToolMode int

const (
	locaToolModeNormal locaToolMode = iota
	locaToolModePTC
)

type locaPromptRunConfig struct {
	maxSteps          int
	singleToolPerTurn bool
	policy            *locaEvidencePolicy
	ptcNullGuard      bool
}

type locaPromptRunResult struct {
	final   any
	hist    []prompt.Prompt
	inTok   int
	outTok  int
	ptcCode string
}

func buildLOCATools(ctx context.Context, mcp *mcpClient, reg *toolRegistry, allowedRoot string, mode locaToolMode, trace *locaTraceCollector, policy *locaEvidencePolicy) (map[string]tools.Tool, []tools.Tool) {
	cache := map[string]any{}
	var cacheMu sync.Mutex
	workspace := newLOCAWorkspaceState(mcp, reg, allowedRoot)
	semantic := newLOCASemanticState()

	toolMap := map[string]tools.Tool{}
	boot := make([]tools.Tool, 0, len(reg.Tools))
	for _, t := range reg.Tools {
		name := t.Name
		serverURL := reg.ToolNameToURL[name]
		orig := reg.ToolNameToOrig[name]
		localTool := t
		localTool.UsePTC = mode == locaToolModePTC
		localTool.Function = func(ctx context.Context, call tools.Call) (string, error) {
			var args map[string]any
			if err := json.Unmarshal(call.Argument, &args); err != nil {
				args = map[string]any{}
			}
			args, err := resolveToolArgsLOCA(ctx, workspace, orig, args)
			if err != nil {
				return "", fmt.Errorf("filesystem path resolution failed before %s: %w", orig, err)
			}
			trace.Add(orig, args)
			policy.ObserveToolAttempt(orig)
			if err := policy.CheckBeforeTool(orig, args); err != nil {
				b, _ := json.Marshal(map[string]any{"error": err.Error()})
				return string(b), nil
			}
			if err := locaValidateCSVWriteLOCA(args, semantic); err != nil {
				b, _ := json.Marshal(map[string]any{"error": err.Error()})
				return string(b), nil
			}

			if isCacheableToolLOCA(orig) {
				key := toolCacheKeyLOCA(serverURL, orig, args)
				cacheMu.Lock()
				cached, ok := cache[key]
				cacheMu.Unlock()
				if ok {
					policy.ObserveTool(orig, args)
					policy.ObserveToolResult(orig, cached, nil)
					semantic.ObserveToolResult(orig, args, cached)
					b, err := json.Marshal(cached)
					if err != nil {
						return "", err
					}
					return string(b), nil
				}
			}

			out, err := mcp.CallTool(ctx, serverURL, orig, args)
			if perr := locaPathResolutionErrorLOCA(err, out); perr != nil && locaToolTouchesFilesystemLOCA(orig, args) {
				refreshedRoot, refreshErr := workspace.Refresh(ctx)
				if refreshErr != nil {
					return "", fmt.Errorf("filesystem path resolution failed for %s: %v (refresh failed: %w)", orig, perr, refreshErr)
				}
				retriedArgs := rewriteCSVPathArgsLOCA(args, refreshedRoot)
				trace.Add(orig, retriedArgs)
				out, err = mcp.CallTool(ctx, serverURL, orig, retriedArgs)
				if retryPathErr := locaPathResolutionErrorLOCA(err, out); retryPathErr != nil {
					return "", fmt.Errorf("filesystem path resolution failed for %s after refresh to %s: %w", orig, refreshedRoot, retryPathErr)
				}
				args = retriedArgs
			}
			if err != nil {
				policy.ObserveToolResult(orig, nil, err)
				return "", err
			}
			policy.ObserveTool(orig, args)
			policy.ObserveToolResult(orig, out, nil)
			semantic.ObserveToolResult(orig, args, out)
			if isCacheableToolLOCA(orig) {
				key := toolCacheKeyLOCA(serverURL, orig, args)
				cacheMu.Lock()
				cache[key] = out
				cacheMu.Unlock()
			}
			b, err := json.Marshal(out)
			if err != nil {
				return "", err
			}
			return string(b), nil
		}
		boot = append(boot, localTool)
		toolMap[localTool.Name] = localTool
	}

	return toolMap, boot
}

func runLOCAPromptLoop(ctx context.Context, g *gen.Generator, hist []prompt.Prompt, cfg locaPromptRunConfig) (locaPromptRunResult, error) {
	resOut := locaPromptRunResult{hist: hist}

	for step := 0; step < cfg.maxSteps; step++ {
		res, err := g.Prompt(resOut.hist...)
		if err != nil {
			return resOut, err
		}
		resOut.inTok += res.Metadata.InputTokens
		resOut.outTok += res.Metadata.OutputTokens

		if res.IsText() {
			if err := cfg.policy.CheckBeforeFinal(); err != nil {
				resOut.hist = append(resOut.hist, prompt.AsUser(locaEvidenceReminderLOCA(err)))
				continue
			}
			text, _ := res.AsText()
			resOut.hist = append(resOut.hist, prompt.AsAssistant(text))
			resOut.final = text
			return resOut, nil
		}
		if !res.IsTools() {
			return resOut, nil
		}
		for i, c := range res.Tools {
			resOut.hist = append(resOut.hist, prompt.AsToolCall(c.ID, c.Name, c.Argument))

			if c.Name == ptc.PTCToolName {
				var arg struct {
					Code string `json:"code"`
				}
				if err := json.Unmarshal(c.Argument, &arg); err == nil {
					resOut.ptcCode = arg.Code
				}
			}

			if cfg.singleToolPerTurn && i > 0 {
				b, _ := json.Marshal(map[string]any{"error": "only one code_execution call allowed per turn"})
				resOut.hist = append(resOut.hist, prompt.AsToolResponse(c.ID, c.Name, string(b)))
				continue
			}

			if c.Ref == nil || c.Ref.Function == nil {
				return resOut, fmt.Errorf("tool %q not found", c.Name)
			}

			out, err := c.Ref.Function(ctx, c)
			if err != nil {
				if locaPathResolutionErrorLOCA(err, nil) != nil {
					return resOut, err
				}
				resOut.hist = append(resOut.hist, prompt.AsToolResponse(c.ID, c.Name, err.Error()))
				continue
			}
			if perr := locaPathResolutionErrorLOCA(nil, out); perr != nil {
				return resOut, fmt.Errorf("filesystem path resolution failed in %s: %w", c.Name, perr)
			}
			out = normalizePTCNullToolResponseLOCA(c.Name, out, cfg.ptcNullGuard)
			resOut.hist = append(resOut.hist, prompt.AsToolResponse(c.ID, c.Name, out))
		}
	}
	return resOut, fmt.Errorf("max steps reached")
}

func runLOCANormal(ctx context.Context, client *bellman.Bellman, mcp *mcpClient, reg *toolRegistry, req locaRequest, allowedRoot string) ([]locaTraceCall, any, []prompt.Prompt, int, int, error) {
	model, err := gen.ToModel(req.BellmanModel)
	if err != nil {
		return nil, nil, nil, 0, 0, err
	}

	trace := &locaTraceCollector{}
	policy := newLOCAEvidencePolicy(req.Query, reg)
	toolMap, boot := buildLOCATools(ctx, mcp, reg, allowedRoot, locaToolModeNormal, trace, policy)

	g := client.Generator().
		Model(model).
		System(req.SystemPrompt).
		SetTools(boot...).
		WithContext(ctx).
		Temperature(req.Temperature)
	if req.MaxTokens > 0 {
		g = g.MaxTokens(req.MaxTokens)
	}
	if tc, err := toolChoiceToConfig(req.ToolChoice, toolMap); err != nil {
		return trace.Snapshot(), nil, nil, 0, 0, err
	} else if tc != nil {
		g = g.SetToolConfig(*tc)
	}

	run, err := runLOCAPromptLoop(ctx, g, []prompt.Prompt{prompt.AsUser(req.Query)}, locaPromptRunConfig{
		maxSteps: 20,
		policy:   policy,
	})
	if err != nil {
		return trace.Snapshot(), nil, run.hist, run.inTok, run.outTok, err
	}
	return trace.Snapshot(), run.final, run.hist, run.inTok, run.outTok, nil
}

func isCacheableToolLOCA(origToolName string) bool {
	// Best-effort heuristic: avoid caching tools that might mutate server state or files.
	n := strings.ToLower(strings.TrimSpace(origToolName))
	if n == "" {
		return false
	}
	// Common mutating verbs.
	mut := []string{"write", "create", "update", "delete", "submit", "enroll", "mark_", "login", "logout", "start_", "publish", "post_"}
	for _, m := range mut {
		if strings.Contains(n, m) {
			return false
		}
	}
	return true
}

func toolCacheKeyLOCA(serverURL, toolName string, args map[string]any) string {
	// encoding/json marshals map keys deterministically.
	b, _ := json.Marshal(args)
	return serverURL + "|" + toolName + "|" + string(b)
}

func runLOCAPTC(ctx context.Context, client *bellman.Bellman, mcp *mcpClient, reg *toolRegistry, req locaRequest, allowedRoot string) ([]locaTraceCall, string, any, []prompt.Prompt, int, int, error) {
	model, err := gen.ToModel(req.BellmanModel)
	if err != nil {
		return nil, "", nil, nil, 0, 0, err
	}

	trace := &locaTraceCollector{}
	policy := newLOCAEvidencePolicy(req.Query, reg)
	toolMap, boot := buildLOCATools(ctx, mcp, reg, allowedRoot, locaToolModePTC, trace, policy)

	g := client.Generator().
		Model(model).
		System(req.SystemPrompt).
		SetTools(boot...).
		WithContext(ctx).
		Temperature(req.Temperature)
	if req.MaxTokens > 0 {
		g = g.MaxTokens(req.MaxTokens)
	}
	g, err = g.ActivatePTC(ptc.JavaScript)
	if err != nil {
		return trace.Snapshot(), "", nil, nil, 0, 0, err
	}
	if tc, err := toolChoiceToConfig(req.ToolChoice, toolMap); err != nil {
		return trace.Snapshot(), "", nil, nil, 0, 0, err
	} else if tc != nil {
		g = g.SetToolConfig(*tc)
	} else {
		g = g.SetToolConfig(tools.AutoTool)
	}

	run, err := runLOCAPromptLoop(ctx, g, []prompt.Prompt{prompt.AsUser(req.Query)}, locaPromptRunConfig{
		maxSteps:          6,
		singleToolPerTurn: true,
		policy:            policy,
		ptcNullGuard:      true,
	})
	if err != nil {
		return trace.Snapshot(), run.ptcCode, nil, run.hist, run.inTok, run.outTok, err
	}
	return trace.Snapshot(), run.ptcCode, run.final, run.hist, run.inTok, run.outTok, nil
}
