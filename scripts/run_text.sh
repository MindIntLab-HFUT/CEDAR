#!/usr/bin/env bash
# Text-only evaluation on CEDAR (7 languages x 1,166 NQ pairs).
# Works with any OpenAI-compatible endpoint, e.g. a local server started by:
#   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
set -e
cd "$(dirname "$0")/.."

BASE_URL=${BASE_URL:-http://localhost:8000/v1}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}

python evaluation/run_benchmark_textonly.py \
  --mode api \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --model_name "$MODEL_NAME" \
  --output_dir results/text-only
