

# NODE_RANK=$1

# export TORCH_HOME=/opt/aps/workdir
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0
export no_proxy=localhost,127.0.0.1


# sudo rm -rf ~/.netrc
#your WANDB_API_KEY here
export WANDB_API_KEY=xxx

# Path of training data
DATA_PATH="../../train/datasets/train_mmlu_medqa.jsonl"

# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH="/AIRPFS/lwt/model/qwen-2.5-7b-instruct"
N_SAMPLES=16
EPISODE=10000
WARMUP=0.0
TBS=256
RBS=64
KL=0.0
LR=2e-6
MAX_LENGTH=29000
PORT=1278
TEMP=1.0

SAVE_MODEL_NAME="qwen-2.5-7b-instruct-ur2-medqa-mmlu"

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p ./trainning_results/$SAVE_MODEL_NAME
mkdir -p ./trainning_results/ckpts
mkdir -p ./trainning_results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

# you can change vllm_tensor_parallel_size from 2 to 4 to accelerate vllm rollout 

ray job submit --address="http://127.0.0.1:8267" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path trainning_results/ckpts/$SAVE_MODEL_NAME \
   --ckpt_path trainning_results/results/ckpts/$SAVE_MODEL_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator reinforce_baseline \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 2048 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --load_checkpoint \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key question \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10\
   --vllm_sync_backend nccl \
   --max_ckpt_num 30 \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME \
   --apply_chat_template \
