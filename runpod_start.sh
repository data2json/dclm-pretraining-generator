#!/bin/bash
# RunPod worker startup script.
# Usage: NUM_SHARDS=4 SHARD_INDEX=0 HF_TOKEN=hf_... bash runpod_start.sh

set -e

NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS required}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
OUTPUT_REPO="${OUTPUT_REPO:-essobi/dclm-crossover-megadoc-p1}"
NUM_GPUS="${NUM_GPUS:-8}"

pip install -q -r requirements.txt

python pipeline.py \
    --output-repo "$OUTPUT_REPO" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --worker-id "w${SHARD_INDEX}" \
    --checkpoint ".checkpoint-w${SHARD_INDEX}.json" \
    --num-gpus "$NUM_GPUS"
