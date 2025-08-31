INPUT_JSON="../dataset/mmlupro/mmlupro.jsonl"  # Replace with the path to the merged MMLU JSON file
SUBJECTS=("economics" "philosophy" "history" "law")  # List of subjects to evaluate    "philosophy" "history"  "law"
SUBSET_NUM=-1  # Number of questions per subject (-1 for all)
TEMPERATURE=0.3
TOP_P=0.5
REPETITION_PENALTY=1.0
MAX_TOKENS=1536
GPU_id="6"
# Convert subjects array to a space-separated string
SUBJECTS_STR=$(IFS=" "; echo "${SUBJECTS[*]}")
MODEL_PATHS=(
    "/AIRPFS/lwt/model/qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.23"
) 
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    python eval_mmlupro_multiturn_search_o1.py \
        --model_path "$MODEL_PATH" \
        --input_json "$INPUT_JSON" \
        --subjects $SUBJECTS_STR \
        --subset_num $SUBSET_NUM \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --repetition_penalty $REPETITION_PENALTY \
        --max_tokens $MAX_TOKENS\
        --gpu_id $GPU_id\
        --gpu_memory_rate 0.9\
        --max_rounds 7
done