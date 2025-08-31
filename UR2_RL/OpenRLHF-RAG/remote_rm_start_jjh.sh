

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
python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT}