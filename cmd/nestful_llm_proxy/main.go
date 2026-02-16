package main

import (
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman"
	nestful "github.com/modfin/bellman/tools/NESTFUL"
)

func main() {
	_ = godotenv.Load(getenvDefault("NESTFUL_ENV", ".env"))

	listen := getenvDefault("NESTFUL_LLM_PROXY_LISTEN", "127.0.0.1:8091")
	upstreamURL := getenvDefault("UPSTREAM_BELLMAN_URL", getenvDefault("BELLMAN_URL", ""))
	upstreamToken := getenvDefault("UPSTREAM_BELLMAN_TOKEN", getenvDefault("BELLMAN_TOKEN", ""))
	upstreamKeyName := getenvDefault("UPSTREAM_BELLMAN_KEY_NAME", getenvDefault("BELLMAN_KEY_NAME", "test"))
	defaultModel := getenvDefault("NESTFUL_MODEL", getenvDefault("BELLMAN_MODEL", "OpenAI/gpt-4o-mini"))

	client := bellman.New(upstreamURL, bellman.Key{Name: upstreamKeyName, Token: upstreamToken})

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	// /nestful: benchmark adapter which returns predicted tool call sequence (with/without PTC)
	mux.HandleFunc("/nestful", nestful.NewNestfulHandler(client, defaultModel))

	log.Printf("NESTFUL LLM proxy listening on http://%s", listen)
	log.Printf("Upstream Bellman URL: %s", upstreamURL)
	log.Printf("Default model: %s", defaultModel)

	if err := http.ListenAndServe(listen, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func getenvDefault(key string, def string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	return v
}
