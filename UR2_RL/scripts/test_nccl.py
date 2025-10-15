# test_nccl.py
import torch
import deepspeed
import argparse

# 添加参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

# 初始化分布式环境前设置CUDA设备
torch.cuda.set_device(args.local_rank)
deepspeed.init_distributed()

rank = torch.distributed.get_rank()
local_rank = args.local_rank

# 显式指定设备创建张量
tensor = torch.tensor([rank], device=torch.device(f'cuda:{local_rank}'))

# AllReduce测试
torch.distributed.all_reduce(tensor)
print(f"Rank {rank}: AllReduce结果 {tensor.item()}")

# Broadcast测试
if rank == 0:
    data = torch.tensor([100], device=torch.device(f'cuda:{local_rank}'))
else:
    data = torch.tensor([0], device=torch.device(f'cuda:{local_rank}'))
torch.distributed.broadcast(data, src=0)
print(f"Rank {rank}: Broadcast结果 {data.item()}")