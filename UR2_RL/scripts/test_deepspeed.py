import torch
import deepspeed
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# 初始化分布式环境
deepspeed.init_distributed(dist_backend="nccl")

# 定义简单模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

# 模型配置
model = SimpleModel().cuda()
ds_config = {
    "train_batch_size": 16,
    "zero_optimization": {
        "stage": 2,
        "reduce_bucket_size": 1e5,
        "allgather_bucket_size": 1e5
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "optimizer": {
        "type": "Adam",       # 或使用 "AdamW", "SGD" 等
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8
        }
    }
}

# 初始化DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# 生成输入数据并转换为bfloat16
data = torch.randn(16, 10).cuda().to(dtype=torch.bfloat16)  # 关键修改：添加数据类型转换

# 执行一次前向+反向传播
loss = model_engine(data).sum()
model_engine.backward(loss)
model_engine.step()

# 打印成功消息
if model_engine.local_rank == 0:
    print("DeepSpeed 初始化测试成功！")