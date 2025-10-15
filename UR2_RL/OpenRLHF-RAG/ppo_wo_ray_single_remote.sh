

JOB_NAME=llama-3-8b-instruct-prompt-collection-v0.1-sample_$N_SAMPLES


NODE_RANK=$1

# export TORCH_HOME=/opt/aps/workdir
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=0f01b18f7114fec603184083ff6b1d5b5ec1983b
# sudo rm -rf ~/.netrc

# Path of training data
DATA_PATH=/home/songhuatong/OpenRLHF/data/rl-0121-short_prompt

# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH=/home/songhuatong/Qwen2.5-0.5B

# export CUDA_VISIBLE_DEVICES=4,5,0,1
N_SAMPLES=8
EPISODE=10000
WARMUP=0.0
TBS=512
RBS=128
KL=0.001
LR=2e-6
MAX_LENGTH=29000
PORT=1278
TEMP=1.0
# REWARD_MODEL=server_false-1_true1_unknown-1-repeat-single
REWARD_MODEL=server_dpsk_tuple
SAVE_MODEL_NAME=final-dpsk1_5b-rm1-1-2-grpo-len_${MAX_LENGTH-}tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-temp$TEMP-30k

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p results/$SAVE_MODEL_NAME
mkdir -p results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/
echo 1111
# pkill -f server_dpsk_tuple
# nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file /home/songhuatong/rm_results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 2>&1 &
# echo $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log

deepspeed \
  --include localhost:0,1,4,5 \
  --module openrlhf.cli.train_ppo \
  --pretrain /home/songhuatong/Qwen2.5-0.5B \
  --remote_rm_url http://localhost:1278/get_reward \
  --save_path /home/songhuatong/RL_Debug/${JOB_NAME} \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 120 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 12 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 2048 \
  --num_episodes 10 \
  --n_samples_per_prompt ${N_SAMPLES} \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.001 \
  --prompt_data /home/songhuatong/OpenRLHF/data/rl-0121-short_prompt \
  --input_key messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --flash_attn \
  --gradient_checkpointing \
  --load_checkpoint \
# &> logs/policy/${JOB_NAME}.log
  # --use_length_reward_in_efficiency \
  # --group_method $GROUP_METHOD \
