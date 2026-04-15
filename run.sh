#!/bin/bash
# Launch pipeline in background with logging to disk.
# Usage: ./run.sh [--limit N] [--dry-run] [extra args...]

set -e

cd "$(dirname "$0")"

# Activate vllm env
source ~/.pyenv/versions/vllm/bin/activate

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/pipeline-$TIMESTAMP.log"
PID_FILE="$LOG_DIR/pipeline.pid"

# Default output repo (override with --output-repo)
OUTPUT_REPO="${OUTPUT_REPO:-essobi/dclm-crossover-megadoc-p1}"

nohup python -u pipeline.py \
    --output-repo "$OUTPUT_REPO" \
    "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Pipeline started (PID $PID)"
echo "Log: $LOG_FILE"
echo ""
echo "Follow log:  tail -f $LOG_FILE"
echo "Stop:        kill \$(cat $PID_FILE)"
echo "Check PID:   ps -p \$(cat $PID_FILE)"
