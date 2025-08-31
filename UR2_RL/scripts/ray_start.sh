# # source /opt/aps/workdir/input/jiechen/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=5,6,1,2,3,4,7,0

# conda activate openrlhf

HEAD_NODE_IP=127.0.0.1


ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 8 --port 8266 --dashboard-port 8267

