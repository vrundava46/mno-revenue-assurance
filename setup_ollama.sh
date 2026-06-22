#!/usr/bin/env bash
# Optional: install Ollama + pull a small local model so the assistant produces
# fluent natural-language answers instead of the deterministic extractive
# fallback. The project works fine WITHOUT this.
set -euo pipefail

MODEL="${TELCO_OLLAMA_MODEL:-llama3.2:3b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found."
  case "$(uname -s)" in
    Darwin) echo "Install it with: brew install ollama   (or download from https://ollama.com/download)";;
    Linux)  echo "Installing via official script..."; curl -fsSL https://ollama.com/install.sh | sh;;
    *)      echo "See https://ollama.com/download";;
  esac
fi

if command -v ollama >/dev/null 2>&1; then
  echo "Starting Ollama server (background) if not already running..."
  (ollama serve >/dev/null 2>&1 &) || true
  sleep 2
  echo "Pulling model: $MODEL"
  ollama pull "$MODEL"
  echo "Done. The assistant will now use Ollama automatically."
fi
