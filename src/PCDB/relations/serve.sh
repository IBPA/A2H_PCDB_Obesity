#!/bin/bash
# Serve the extraction model with the configuration the published labels used.
#
# One server per GPU, tensor-parallel-size 1: the model fits on a single 24 GB
# card, so N independent servers give ~N x throughput where TP=N would add
# collective overhead and reduce it. Run this once per GPU with a distinct
# GPU_ID and PORT, then point one run_extraction.py at each.
#
# MAXLEN is the lever that matters. Prompt and completion share the context
# window, and extraction needs room to reason over a whole paper: at 20480 the
# longest prompts left too little for the answer and verdicts came back
# missing. Raise MAXLEN for those, do not raise --max-tokens.
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-20480}"
GPU_FRAC="${GPU_FRAC:-0.90}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_FRAC" \
  --reasoning-parser gemma4 \
  --host 127.0.0.1 --port "$PORT"
