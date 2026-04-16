#!/bin/bash
set -e # Exit immediately if a command fails

echo "--- Starting Swarm Setup ---"

# 1. Install dependencies
pip install hf_transfer datasets huggingface_hub pyarrow tqdm "vllm<=0.18.0"
# 2. Clone the repo (if not already there)
git clone https://github.com/data2json/dclm-pretraining-generator.git /workspace/repo || true
cd /workspace/repo

# 3. Handle HF Login
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
fi

# 4. START THE WORKER (This keeps the pod alive)
echo "--- Launching Worker $RUNPOD_SHARD_INDEX ---"
python pipeline.py \
    --output-repo "$OUTPUT_REPO" \
    --num-shards "${TOTAL_SHARDS:-1}" \
    --shard-index "${RUNPOD_SHARD_INDEX:-0}" \
    --worker-id "pod-${RUNPOD_SHARD_INDEX:-0}"
