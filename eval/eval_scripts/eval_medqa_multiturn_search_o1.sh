INPUT_JSON="../mmlupro_med/mmlupro_med.jsonl"  # you can change this path to "../dataset/medqa/medqa_500_5options.jsonl" to test medqa(5options)
SUBSET_NUM=-1  # 评估的问题数量（-1 表示全部）
TEMPERATURE=0.3
TOP_P=0.5
REPETITION_PENALTY=1.0
MAX_TOKENS=1536
GPU_id="0,7"
TOPK=10
MODEL_PATHS=(
    "/AIRPFS/lwt/model/qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.23"
) 
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    python eval_medqa_multiturn_search_o1.py \
        --model_path "$MODEL_PATH" \
        --input_json "$INPUT_JSON" \
        --subset_num $SUBSET_NUM \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --repetition_penalty $REPETITION_PENALTY \
        --max_tokens $MAX_TOKENS \
        --gpu_id $GPU_id \
        --gpu_memory_rate 0.9 \
        --topk $TOPK\
        --max_rounds 7
done