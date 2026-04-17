#!/bin/bash
set -e # Exit immediately if a command fails

echo "--- Starting Swarm Setup ---"
pip freeze | grep -i torch > preserve.txt
# 1. Install dependencies
pip install hf_transfer datasets huggingface_hub pyarrow tqdm arctic-inference==0.1.1 https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl vllm -c preserve.txt
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
