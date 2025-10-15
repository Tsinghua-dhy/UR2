import threading
import time
from openai import OpenAI
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

#replace your api key here
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
def GPT4omini_batch_request(prompts, max_threads=128):
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

def eval_math_scores(queries, answers, preds, sources, is_equivs):
    scores = []
    gpt_eval_indices = []
    gpt_prompts = []

    # 先挑出需要用 GPT 评估的
    for i in range(len(queries)):
        if sources[i] == 'math' and not is_equivs[i]:
            gpt_eval_indices.append(i)
            gpt_prompts.append(get_prompt(queries[i], answers[i], preds[i]))

    # 批量调用 GPT4o-mini 进行评估
    gpt_outputs = GPT4omini_batch_request(gpt_prompts)

    # 映射 GPT 评估结果
    gpt_judgments = []
    for output in gpt_outputs:
        match = re.search(r"Judgment:\s*(Correct|Partially correct|Incorrect)", output)
        judgment = match.group(1) if match else "Incorrect"
        score = {"Correct": 2.0, "Partially correct": 1.0, "Incorrect": 0.0}.get(judgment, 0.0)
        gpt_judgments.append(score)

    # 构造最终 scores 列表
    gpt_ptr = 0
    for i in range(len(queries)):
        if sources[i] == 'math':
            if is_equivs[i]:
                scores.append(2.0)
            else:
                scores.append(gpt_judgments[gpt_ptr])
                gpt_ptr += 1
        else:
            # 对于非 math 类型，打分规则留给外部函数决定
            scores.append(None)

    return scores
