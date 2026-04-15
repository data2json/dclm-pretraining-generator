#!/bin/bash
# start-smollm2-fleet.sh

for gpu_id in 0 1 2 3 4 5 6 7; do
    port=$((8080 + gpu_id))
    docker run -d \
        --runtime nvidia \
        --gpus "\"device=${gpu_id}\"" \
        -v /mnt/fivebucks/public/ai-share/vllm/huggingface:/root/.cache/huggingface \
        --rm \
        --name smollm2-gpu${gpu_id} \
        --env HUGGING_FACE_HUB_TOKEN=hf_TOKEN \
        --env CUDA_VISIBLE_DEVICES=0 \
        --dns 8.8.8.8 --dns 8.8.4.4 \
        --network=host \
        --ipc=host \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864:67108864 \
        --ulimit nofile=65536:65536 \
        vllm/vllm-openai:v0.8.5 \
        --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --served-model-name 'gpt-3.5-turbo' \
        --gpu-memory-utilization 0.50 \
        --max-log-len 10 \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --dtype=half \
        --port ${port} \
        --max-model-len 4096 \
        --max-num-seqs 256 \
        --max-num-batched-tokens 4096 \
        --enable-prefix-caching \
        --enforce-eager
done

echo "All containers launched, waiting for health..."

for gpu_id in 0 1 2 3 4 5 6 7; do
    port=$((8080 + gpu_id))
    retries=0
    until curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        sleep 2
        retries=$((retries + 1))
        if [ $retries -ge 60 ]; then
            echo "TIMEOUT: GPU ${gpu_id} on port ${port} failed to start"
            docker logs smollm2-gpu${gpu_id} --tail 20
            break
        fi
    done
    if [ $retries -lt 60 ]; then
        echo "GPU ${gpu_id} ready on port ${port}"
    fi
done

echo ""
echo "Fleet status:"
docker ps --filter "name=smollm2-gpu" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Stop with: for i in 0 1 2 3 4 5 6 7; do docker stop smollm2-gpu\$i; done"
