#!/bin/bash
# Usage: HF_TOKEN=hf_... bash launch.sh
# Or set HF_TOKEN in the RunPod pod environment variables.

set -e

: "${HF_TOKEN:?HF_TOKEN must be set}"

pkill -f pipeline.py 2>/dev/null || true
cd /workspace/dclm-pretraining-generator
git pull

tmux new-session -d -s pipeline \
  "env HF_TOKEN=$HF_TOKEN NUM_SHARDS=${NUM_SHARDS:-1} SHARD_INDEX=${SHARD_INDEX:-0} OUTPUT_REPO=${OUTPUT_REPO:-essobi/dclm-crossover-megadoc-p1} NUM_GPUS=${NUM_GPUS:-8} bash runpod_start.sh 2>&1 | tee /workspace/pipeline.log"

echo "Pipeline launched in tmux session 'pipeline'"
echo "Attach with: tmux attach -t pipeline"
echo "Tail log:    tail -f /workspace/pipeline.log"
