#!/usr/bin/env bash
# Multimodal evaluation on CEDAR (7 languages x 400 image-narrative-question triples).
# Works with any OpenAI-compatible endpoint that accepts images, e.g.:
#   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
set -e
cd "$(dirname "$0")/.."

BASE_URL=${BASE_URL:-http://localhost:8000/v1}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}

python evaluation/run_benchmark_multimodal.py \
  --mode api \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --model_name "$MODEL_NAME" \
  --output_dir results/multimodal
