# source /opt/aps/workdir/input/jiechen/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# conda activate openrlhf

HEAD_NODE_IP=183.174.229.142

ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 10 --port 8266 --dashboard-port 8267


# NODE_IP=$(hostname -I | awk '{print $1}')
# ray start --address ${HEAD_NODE_IP}:6379 --num-gpus 8 --node-ip-address $NODE_IP