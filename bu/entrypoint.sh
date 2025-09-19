#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI互換APIサーバを起動。compose の command: がそのまま引数に入る
exec python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
