import ray
from vllm import LLM, SamplingParams


# 定义一个 Ray 远程函数，用于执行 vLLM 推理任务
@ray.remote(num_gpus=1)
def generate_text(prompt):
    # 初始化 vLLM 的 LLM 对象，这里以 Vicuna 模型为例，你可以替换为自己想用的模型
    llm = LLM(model="/home/songhuatong/Qwen2-1.5B-Instruct")
    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # 进行文本生成
    result = llm.generate(prompt, sampling_params)
    return result[0].outputs[0].text


if __name__ == "__main__":
    try:
        # 连接到已启动的 Ray 集群
        ray.init(address='auto')

        # 定义多个要生成文本的提示
        prompts = [
            "介绍一下北京故宫",
            "简述太阳系的结构",
            "讲讲唐朝的文化"
        ]

        # 并行提交多个任务到 Ray
        result_ids = [generate_text.remote(prompt) for prompt in prompts]

        # 获取所有任务的结果
        results = ray.get(result_ids)

        # 打印每个任务的结果
        for i, result in enumerate(results):
            print(f"提示 {i + 1} 生成的文本：")
            print(result)
            print("-" * 50)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
    finally:
        # 关闭 Ray
        ray.shutdown()