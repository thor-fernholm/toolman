package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/joho/godotenv"
	"github.com/modfin/bellman/tools/ptc/bfcl"
	"github.com/modfin/bellman/tools/ptc/cfb"
)

func main() {
	// godotenv.Load() ...
	err := godotenv.Load()
	if err != nil {
		panic(err)
	}

	// Register API Endpoint
	http.HandleFunc("/bfcl", bfcl.MiddlewareDebugLogger("BFCL", bfcl.HandleGenerateBFCL))
	http.HandleFunc("/cfb", bfcl.MiddlewareDebugLogger("CFB", cfb.HandleGenerateCFB))

	// Register Debug UI Endpoints
	http.HandleFunc("/debug", bfcl.HandleDebugUI)
	http.HandleFunc("/debug/api/data", bfcl.HandleDebugData)
	http.HandleFunc("/debug/api/clear", bfcl.HandleDebugClear)

	fmt.Println("---------------------------------------------------------")
	fmt.Println(" Toolman Bench Server Running")
	fmt.Println(" BFCL API Endpoint:   http://localhost:8080/bfcl")
	fmt.Println(" CFB API Endpoint:    http://localhost:8080/cfb")
	fmt.Println(" BFCL Debug UI:       http://localhost:8080/debug")
	fmt.Println("---------------------------------------------------------")

	fmt.Println("Toolman Benchmark Server running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
