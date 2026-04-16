export OUTPUT_REPO="essobi/dclm-crossover-megadoc-p1"

for i in {0..9}; do
  sky jobs launch -n "megadoc-shard-$i" megadoc_4090.yaml \
    --env SHARD_IDX=$i \
    --env TOTAL_SHARDS=10 \
    --env HF_TOKEN=$HF_TOKEN \
    --env OUTPUT_REPO=$OUTPUT_REPO \
    --disk-size 40 \
    --detach-run -y
done
