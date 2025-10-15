

# NODE_RANK=$1

# export TORCH_HOME=/opt/aps/workdir
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=0f01b18f7114fec603184083ff6b1d5b5ec1983b
# sudo rm -rf ~/.netrc
export WANDB_API_KEY=0f01b18f7114fec603184083ff6b1d5b5ec1983b

# Path of training data
DATA_PATH=/home/songhuatong/RL_Debug/data_rollout_10/used_front_2k.jsonl
# /home/songhuatong/OpenRLHF/data/demo_dataset
#
# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH=/home/songhuatong/Qwen2.5-1.5B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
N_SAMPLES=16
EPISODE=10000
WARMUP=0.0
TBS=64
RBS=16
KL=0.0005
LR=2e-6
MAX_LENGTH=29000
PORT=1278
TEMP=1.0
# REWARD_MODEL=server_false-1_true1_unknown-1-repeat-single
REWARD_MODEL=server_dpsk_tuple
SAVE_MODEL_NAME=new-qwen2.5-1.5B-rm1-1-2-grpo-len_${MAX_LENGTH-}tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-temp$TEMP-30k

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p /home/songhuatong/RAG_RL/results/$SAVE_MODEL_NAME
mkdir -p /home/songhuatong/RAG_RL/results/ckpts
mkdir -p /home/songhuatong/RAG_RL/results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

# pkill -f ${REWARD_MODEL}
# nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file /home/songhuatong/RAG_RL/results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 2>&1 &
# echo $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log

ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json='{"working_dir": "/home/songhuatong/OpenRLHF", "RAY_DEDUP_LOGS": 0}' \
   -- /home/songhuatong/miniconda3/envs/openrlhf/bin/python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path /home/songhuatong/RAG_RL/results/ckpts/$SAVE_MODEL_NAME \
   --ckpt_path /home/songhuatong/RAG_RL/results/ckpts/$SAVE_MODEL_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 1024 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key question \
   --apply_chat_template \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 25 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 3 \
   --group_method $GROUP_METHOD \
   --use_length_reward_in_efficiency \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME \
