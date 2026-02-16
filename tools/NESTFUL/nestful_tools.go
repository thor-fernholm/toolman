package nestful

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/modfin/bellman/models/gen"
)

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func httpErr(w http.ResponseWriter, err error, status int) {
	writeJSON(w, status, map[string]any{"error": err.Error()})
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func parseModelFQN(fqn string) (gen.Model, error) {
	fqn = strings.TrimSpace(fqn)
	provider, name, found := strings.Cut(fqn, "/")
	if !found {
		provider, name, found = strings.Cut(fqn, ".")
	}
	if !found {
		return gen.Model{}, fmt.Errorf("expected provider/name (or provider.name), got %q", fqn)
	}
	provider = canonicalProvider(provider)
	name = canonicalModelName(name)
	if provider == "" || name == "" {
		return gen.Model{}, fmt.Errorf("expected provider/name (or provider.name), got %q", fqn)
	}
	return gen.Model{Provider: provider, Name: name}, nil
}
func canonicalProvider(p string) string {
	pl := strings.ToLower(strings.TrimSpace(p))
	switch pl {
	case "openai":
		return "OpenAI"
	case "vertexai", "vertex":
		return "VertexAI"
	case "anthropic":
		return "Anthropic"
	case "ollama":
		return "Ollama"
	case "vllm":
		return "vLLM"
	case "voyageai", "voyage":
		return "VoyageAI"
	default:
		return strings.TrimSpace(p)
	}
}

func canonicalModelName(n string) string {
	n = strings.TrimSpace(n)
	n = strings.ReplaceAll(n, "_", "-")
	if strings.HasPrefix(n, "gpt4o-") {
		n = "gpt-4o-" + strings.TrimPrefix(n, "gpt4o-")
	}
	return n
}
