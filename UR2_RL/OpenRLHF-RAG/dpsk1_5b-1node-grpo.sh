

NODE_RANK=$1

# export TORCH_HOME=/opt/aps/workdir
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=0f01b18f7114fec603184083ff6b1d5b5ec1983b
# sudo rm -rf ~/.netrc

# Path of training data
DATA_PATH=/home/songhuatong/OpenRLHF/data/demo_dataset

# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH=/home/songhuatong/Qwen2-1.5B-Instruct

export CUDA_VISIBLE_DEVICES=6,7,8,9
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

pkill -f ${REWARD_MODEL}
nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 2>&1 &
echo $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log

if [ "$NODE_RANK" = "0" ]; then
ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json='{"working_dir": "/home/songhuatong/OpenRLHF", "RAY_DEDUP_LOGS": 0, "OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1}' \
   -- /home/songhuatong/miniconda3/envs/openrlhf/bin/python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path results/$SAVE_MODEL_NAME \
   --ckpt_path results/$SAVE_MODEL_NAME \
   --micro_train_batch_size 1 \
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
   --input_key messages \
   --apply_chat_template \
   --packing_samples \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --use_wandb ${wandb_token} \
   --wandb_org xxx \
   --wandb_run_name $SAVE_MODEL_NAME \
   --wandb_project zzz \
   --vllm_sync_backend nccl \
   --max_ckpt_num 20 \
   --group_method $GROUP_METHOD \
   --use_length_reward_in_efficiency \
   --temperature $TEMP \
   --overlap_comm
fi
#    --enable_ema \
#    --load_checkpoint

# python /home/songhuatong/OpenRLHF/openrlhf/cli/server_dpsk_tuple.py --data_path /home/songhuatong/OpenRLHF/data/rl-0121-short_prompt --reward_pretrain /home/songhuatong/Qwen2.5-0.5B --log_file /home/songhuatong/OpenRLHF/data/sampling.jsonl --port 1278
