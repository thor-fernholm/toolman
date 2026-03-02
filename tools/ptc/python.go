package ptc

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/modfin/bellman/tools"
)

// PythonRuntime holds the registered tools
type PythonRuntime struct {
	Tools map[string]tools.Tool
}

type ToolRequest struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args"`
}

// adaptToolsToPythonPTC converts Bellman tools into a Python Docker PTC tool
func adaptToolsToPythonPTC(runtime *PythonRuntime, inputTools []tools.Tool) (tools.Tool, string, error) {
	runtime.Tools = make(map[string]tools.Tool)
	var descriptions []string
	var shimBuilder strings.Builder

	// The Python Shim: Standard library HTTP client over Unix Sockets
	shimBuilder.WriteString(`import json, http.client, socket

class UnixHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.host)

def __call_go_tool(tool_name, args):
    try:
        conn = UnixHTTPConnection("/workspace/agent.sock")
        payload = json.dumps({"tool": tool_name, "args": args})
        conn.request("POST", "/execute", body=payload, headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        data = res.read().decode()
        conn.close()
        return json.loads(data)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}

`)

	// Register tools and build Python wrappers
	for _, t := range inputTools {
		runtime.Tools[t.Name] = t
		descriptions = append(descriptions, formatToolSignature(t))

		shim := fmt.Sprintf(`def %s(args):
    return __call_go_tool("%s", args)
`, t.Name, t.Name)
		shimBuilder.WriteString(shim + "\n")
	}

	pythonShim := shimBuilder.String()

	type CodeArgs struct {
		Code string `json:"code" json-description:"The executable top-level Python code string."`
	}

	// The Execution Function
	executor := func(ctx context.Context, call tools.Call) (string, error) {
		var arg CodeArgs
		if err := json.Unmarshal(call.Argument, &arg); err != nil {
			return "", err
		}

		// Create a temporary workspace for this specific execution
		workspace, err := os.MkdirTemp("", "ptc-agent-*")
		if err != nil {
			return "", fmt.Errorf("failed to create workspace: %w", err)
		}
		defer os.RemoveAll(workspace) // Cleanup after execution

		sockPath := filepath.Join(workspace, "agent.sock")
		scriptPath := filepath.Join(workspace, "script.py")

		// Write the combined Python script
		fullCode := pythonShim + "\n\n# --- LLM CODE --- \n" + arg.Code
		if err := os.WriteFile(scriptPath, []byte(fullCode), 0644); err != nil {
			return "", err
		}

		// Start the Unix Socket HTTP Server
		listener, err := net.Listen("unix", sockPath)
		if err != nil {
			return "", fmt.Errorf("failed to create unix socket: %w", err)
		}
		defer listener.Close()

		mux := http.NewServeMux()
		mux.HandleFunc("/execute", func(w http.ResponseWriter, r *http.Request) {
			var req ToolRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, `{"error": "invalid payload"}`, http.StatusBadRequest)
				return
			}

			tool, exists := runtime.Tools[req.Tool]
			if !exists {
				w.Write([]byte(`{"error": "tool not found"}`))
				return
			}

			// Execute the Go tool
			resStr, err := tool.Function(ctx, tools.Call{Argument: req.Args})
			if err != nil {
				w.Write([]byte(fmt.Sprintf(`{"error": %q}`, err.Error())))
				return
			}
			w.Write([]byte(resStr))
		})

		server := &http.Server{Handler: mux}
		go server.Serve(listener)
		defer server.Close()

		// TODO fix workaround?
		if ctx == nil {
			ctx = context.Background()
		}
		// Configure the Docker Sandbox
		execCtx, cancel := context.WithTimeout(ctx, 20*time.Second) // 20s max execution
		defer cancel()

		cmd := exec.CommandContext(execCtx, "docker", "run",
			"--rm",              // Delete when done
			"--network", "none", // No internet access
			"--cap-drop", "ALL", // Drop Linux capabilities
			"--security-opt", "no-new-privileges", // Prevent privilege escalation
			"--memory", "256m", // RAM limit
			"-v", fmt.Sprintf("%s:/workspace", workspace), // Mount the workspace
			"-w", "/workspace", // Set working directory
			"python:3.11-alpine", // Lightweight image
			"python", "script.py",
		)

		// Execute and Capture Output
		output, err := cmd.CombinedOutput()

		if err != nil {
			if execCtx.Err() == context.DeadlineExceeded {
				return `{"error": "timeout: python script took too long"}`, nil
			}
			// Return output (which contains Python traceback) so LLM can self-correct
			return fmt.Sprintf(`{"error": %q, "traceback": %q}`, err.Error(), string(output)), nil
		}

		outStr := strings.TrimSpace(string(output))
		if outStr == "" {
			return "null", nil
		}
		return outStr, nil
	}

	docsFragment := strings.Join(descriptions, "\n\n")

	// Create the final Tool
	ptcTool := tools.NewTool("PythonCodeExecution",
		tools.WithDescription(`Execute top-level Python 3 code in a sandboxed environment to call available Tool Functions.

RULES:
- At most ONE script per turn. Write a complete batch script.
- Python indentation is strictly enforced.
- To return data to the system, you MUST print() it as a JSON string on the final line.
- Example: print(json.dumps({"result": fetch_users({"limit": 5})}))

Available Python Tool Functions:`+
			"\n\n"+docsFragment,
		),
		tools.WithArgSchema(CodeArgs{}),
		tools.WithFunction(executor),
	)

	systemFragment := "\n\n## Available Python Tool Functions:\n\n" + docsFragment

	return ptcTool, systemFragment, nil
}
