#!/bin/bash

models1=(
    "/AIRPFS/lwt/model/qwen-2.5-7b-instruct-rl-ours-v3.0.8-stage2"
)
gpu_id="7"
#you can change src_file to "../minervamath/minervamath.jsonl" to test minervamath.jsonl
for model_path in "${models1[@]}"; do
    python ./eval_math500_multiturn.py \
        --gpu_id $gpu_id \
        --temp 0.3 \
        --port 5006 \
        --src_file "../dataset/math500/test.jsonl"\
        --model_path "${model_path}" \
        --topk 10\
        --gpu_memory_rate 0.95
done



