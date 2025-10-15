#!/bin/bash

model_path="/AIRPFS/lwt/model/qwen-2.5-7b-instruct-rl-ours-v3.0.8-stage2"
gpu_id="7"

# 定义 src_file 列表
src_files=(
    "../dataset/bamboogle/bamboogle.jsonl"
    "../dataset/hotpot500/hotpotqa_500.jsonl"
    "../dataset/musique500/musique_500.jsonl"
    "../dataset/2wiki500/2wiki_500.jsonl"
)

# 遍历 src_file 列表并执行命令
for src_file in "${src_files[@]}"; do
    python ./eval_hotpot500_multiturn_search_o1.py \
        --gpu_id $gpu_id \
        --temp 0.3 \
        --top_p 0.5 \
        --port 5006 \
        --src_file "$src_file" \
        --model_path "${model_path}" \
        --gpu_memory_rate 0.95 \
        --topk 10
done