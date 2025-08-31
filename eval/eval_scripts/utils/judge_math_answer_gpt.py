import json
import asyncio
from openai import AsyncOpenAI
import re
import httpx
import threading
import time
from openai import OpenAI
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
from typing import List, Dict, Optional
"""
api_4_key = ""

# 创建同步 OpenAI 客户端
client_4 = OpenAI(
    base_url="https://svip.xty.app/v1",
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url="https://svip.xty.app/v1",
        follow_redirects=True,
    ),
)

"""

## replace your api key here
api_4_key = ""

# 创建同步 OpenAI 客户端
client_4 = OpenAI(
    base_url="https://api.key77qiqi.com/v1",
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url="https://api.key77qiqi.com/v1",
        follow_redirects=True,
    ),
)

# 单个请求函数
def gpt4omini_request(prompt):
    while True:
        try:
            rst = client_4.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.0,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return rst.choices[0].message.content.strip()
        except Exception as e:
            print("ChatGPT ERROR:", e)
            time.sleep(1)

# 批量请求函数（用线程池，并发上限控制在 128）
def GPT4omini_batch_request(prompts, max_threads=256):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_idx = {
            executor.submit(gpt4omini_request, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Task {idx} failed:", e)
    return results

def get_prompt(question, gold, pred):
    return f"""You are an expert math evaluator.
Given a question, a gold answer and a predicted answer, judge if they are mathematically consistent.

Ignore formatting (e.g., \\text{{}}, spacing, capitalization).
Accept equivalent expressions (e.g., factored vs expanded form).
If the prediction matches only part of a multi-part answer (e.g. one of several intervals or roots), label it as Partially correct.

Output format:
Reason: Brief explanation
Judgment: Correct / Partially correct / Incorrect

Input:
Question: {question}
Gold: {gold}
Pred: {pred}"""


def eval_gpt(data):
    # 先挑出需要评估的
    data_to_eval = [item for item in data if item["Metrics"]["math_equal"] == False]
    prompts = [get_prompt(item['question'], item['answer'], item['pred_ans']) for item in data_to_eval]
    outputs = GPT4omini_batch_request(prompts)

    for item, eval_output in zip(data_to_eval, outputs):
        item['eval'] = eval_output
        match = re.search(r"Judgment:\s*(Correct|Partially correct|Incorrect)", eval_output)
        judgment = match.group(1) if match else "Incorrect"
        tag = {"Correct": 1, "Partially correct": 0.5, "Incorrect": 0}.get(judgment, 0)
        item["Metrics"]['math_equal_gpt'] = tag

    # 对所有数据，补全 math_equal_gpt 字段，默认 1
    for item in data:
        if 'math_equal_gpt' not in item["Metrics"]:
            item["Metrics"]['math_equal_gpt'] = 1

    return data

def eval_qwen3(data: List[dict], url: str = "http://localhost:8079/eval_math_scores"):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 抛出 HTTP 错误
        return response.json()["data"]
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None