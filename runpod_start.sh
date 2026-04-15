#!/bin/bash
# RunPod worker startup script.
# Usage: NUM_SHARDS=4 SHARD_INDEX=0 HF_TOKEN=hf_... bash runpod_start.sh

set -e

NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS required}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
OUTPUT_REPO="${OUTPUT_REPO:-essobi/dclm-crossover-megadoc-p1}"
NUM_GPUS="${NUM_GPUS:-8}"

export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

apt-get install -y tmux -qq 2>/dev/null || true
pip install -q -r requirements.txt

# Install runpodctl for self-termination
if ! command -v runpodctl &>/dev/null; then
    curl -fsSL https://cli.runpod.io/install.sh | sh 2>/dev/null || true
fi

python pipeline.py \
    --output-repo "$OUTPUT_REPO" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --worker-id "w${SHARD_INDEX}" \
    --checkpoint ".checkpoint-w${SHARD_INDEX}.json" \
    --num-gpus "$NUM_GPUS"

echo "Pipeline complete. Stopping pod ${RUNPOD_POD_ID}..."
runpodctl pod stop "$RUNPOD_POD_ID"
