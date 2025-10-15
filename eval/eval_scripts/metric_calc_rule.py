"""
update from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py and https://github.com/mandarjoshi90/triviaqa/blob/master/triviaqa_evaluation.py
"""

import pdb
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
import jsonlines
import time
import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )


def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    # print("prediction: ",prediction)
    # print("ground_truth: ",ground_truth)
    # print("pre_list: ",pre_list)
    # print("ground_list: ",ground_list)
    # 不考虑顺序和连续
    return all(ground in pre_list for ground in ground_list)


def cover_exact_match_score_2(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True
    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)
    if ground_str in pre_str:
        return True
    return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if metric_fn.__name__ == "exact_match_score":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, prec, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, prec, recall))
        f1, prec, recall = max(scores_for_ground_truths, key=lambda x: x[0])
        return f1, prec, recall
    elif metric_fn.__name__ == "cover_exact_match_score_1":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "cover_exact_match_score_2":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


# LLM as Judge 相关函数

#your openai api here for llm-as-a-judge
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

def get_judge_prompt(question, gold_answer, predicted_answer):
    """生成用于LLM判断的prompt"""
    return f"""Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.

Question: {question}
Golden Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Your response should be exactly "True" or "False"."""

def gpt4omini_request(prompt):
    """单个GPT请求"""
    while True:
        try:
            rst = client_4.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.0,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}]
            )
            return rst.choices[0].message.content.strip()
        except Exception as e:
            print("ChatGPT ERROR:", e)
            time.sleep(1)

def GPT4omini_batch_request(prompts, max_threads=128):
    """批量GPT请求"""
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
                results[idx] = "False"  # 默认为False
    return results

def llm_as_judge_score(question, prediction, ground_truth):
    """使用LLM作为评判器评分"""
    prompt = get_judge_prompt(question, ground_truth, prediction)
    result = gpt4omini_request(prompt)
    # 解析结果，返回布尔值
    return result.lower().strip() == "true"

def update_answer_with_llm_judge(metrics, question, prediction, gold, use_llm_judge=False):
    """更新答案评估结果，包含LLM judge选项"""
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold)
    f1, prec, recall = metric_max_over_ground_truths(f1_score, prediction, gold)
    cover_em_1 = metric_max_over_ground_truths(
        cover_exact_match_score_1, prediction, gold
    )
    cover_em_2 = metric_max_over_ground_truths(
        cover_exact_match_score_2, prediction, gold
    )

    # LLM as Judge评分 - 只有当F1不是1.0时才使用
    llm_judge_score_val = 0
    if use_llm_judge and f1 < 1.0:
        # 对于多个ground truth，取最高分
        llm_scores = []
        for gt in gold:
            score = llm_as_judge_score(question, prediction, gt)
            llm_scores.append(score)
        llm_judge_score_val = max(llm_scores) if llm_scores else 0
    elif use_llm_judge and f1 == 1.0:
        # F1为1.0时，直接设为1
        llm_judge_score_val = 1

    metrics["em"] += float(em)
    metrics["cover_em_1"] += float(cover_em_1)
    metrics["cover_em_2"] += float(cover_em_2)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall
    if use_llm_judge:
        metrics["llm_judge"] += float(llm_judge_score_val)

    if cover_em_1:
        metrics["acc_num"] += 1
    
    return em, prec, recall, f1, cover_em_1, cover_em_2, llm_judge_score_val

def batch_llm_as_judge_evaluation(data_items):
    """批量LLM评估 - 只评估F1 < 1.0的样本"""
    # 先计算所有样本的F1分数，筛选出需要LLM评估的样本
    items_to_eval = []
    item_indices = []
    
    for i, item in enumerate(data_items):
        prediction = item["pred_ans"]
        if isinstance(item["answer"], list):
            gold_answers = item["answer"]
        else:
            gold_answers = [item["answer"]]
        
        # 计算F1分数
        f1, _, _ = metric_max_over_ground_truths(f1_score, prediction, gold_answers)
        
        if f1 < 1.0:  # 只有F1不是1.0的才需要LLM评估
            items_to_eval.append(item)
            item_indices.append(i)
    
    print(f"LLM Judge: Evaluating {len(items_to_eval)} out of {len(data_items)} samples (F1 < 1.0)")
    
    if not items_to_eval:
        # 如果没有需要评估的样本，返回全1的结果
        return [1.0] * len(data_items)
    
    # 为需要评估的样本生成prompts
    prompts = []
    for item in items_to_eval:
        question = item["question"]
        prediction = item["pred_ans"]
        
        # 处理多个ground truth的情况
        if isinstance(item["answer"], list):
            gold_answers = item["answer"]
        else:
            gold_answers = [item["answer"]]
        
        # 为每个ground truth生成prompt
        for gold_answer in gold_answers:
            prompt = get_judge_prompt(question, gold_answer, prediction)
            prompts.append((prompt, len(prompts) // len(gold_answers)))  # 记录对应的item索引
    
    # 批量请求
    prompt_strs = [p[0] for p in prompts]
    results = GPT4omini_batch_request(prompt_strs)
    
    # 整理结果
    llm_scores = {}
    for i, (prompt_info, result) in enumerate(zip(prompts, results)):
        item_idx = prompt_info[1]
        if item_idx not in llm_scores:
            llm_scores[item_idx] = []
        llm_scores[item_idx].append(result.lower().strip() == "true")
    
    # 构建完整的结果数组
    final_scores = []
    eval_idx = 0
    for i in range(len(data_items)):
        if i in item_indices:  # 这个样本需要LLM评估
            if eval_idx in llm_scores:
                final_scores.append(max(llm_scores[eval_idx]))
            else:
                final_scores.append(False)
            eval_idx += 1
        else:  # F1 = 1.0的样本，直接设为True
            final_scores.append(True)
    
    return final_scores

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data

def eval(file, use_llm_judge=True, batch_llm_eval=True):
    data = read_jsonl(file)[:2000]
    print(len(data))
    print(f"Eval {len(data)} from {file}")
    
    metrics = {
        "em": 0,
        "f1": 0,
        "cover_em_1": 0,
        "cover_em_2": 0,
        "prec": 0,
        "recall": 0,
        "acc_num": 0
    }
    
    if use_llm_judge:
        metrics["llm_judge"] = 0
    
    # 如果使用批量LLM评估
    if use_llm_judge and batch_llm_eval:
        print("Using batch LLM evaluation...")
        llm_judge_scores = batch_llm_as_judge_evaluation(data)
    
    for i, d in enumerate(data):
        pred_answer = d["pred_ans"]
        question = d.get("question", "")

        if isinstance(d["answer"], list):
            gold_answers = d["answer"]
        else:
            gold_answers = [d["answer"]]
        
        if use_llm_judge and not batch_llm_eval:
            # 单个评估模式
            em, prec, recall, f1, cover_em_1, cover_em_2, llm_judge_score_val = update_answer_with_llm_judge(
                metrics, question, pred_answer, gold_answers, use_llm_judge=True
            )
        elif use_llm_judge and batch_llm_eval:
            # 批量评估模式，直接使用预计算的结果
            em, prec, recall, f1, cover_em_1, cover_em_2, _ = update_answer_with_llm_judge(
                metrics, question, pred_answer, gold_answers, use_llm_judge=False
            )
            llm_judge_score_val = llm_judge_scores[i]
            metrics["llm_judge"] += float(llm_judge_score_val)
            
            # 添加调试信息
            if f1 < 1.0 and llm_judge_score_val:
                print(f"Sample {i}: F1={f1:.3f}, LLM-Judge=True - Rescued by LLM!")
        else:
            # 不使用LLM评估
            em, prec, recall, f1, cover_em_1, cover_em_2, llm_judge_score_val = update_answer_with_llm_judge(
                metrics, question, pred_answer, gold_answers, use_llm_judge=False
            )
            llm_judge_score_val = 0

        # 可选：打印不匹配的案例（调试用）
        # if not cover_em_2:
        #     if d.get("gpt4o_output") == "True":
        #         print("==="*40)
        #         print("ques:", d["question"])
        #         print("pred:", pred_answer)
        #         print("gold:", d["answer"])
        #         print(f"f1:{f1}, cover_em_1:{cover_em_1}, llm_judge:{llm_judge_score_val}")
        #         print("==="*40)

    N = len(data)
    for k in metrics.keys():
        if k == "acc_num":
            continue
        metrics[k] /= N

    # 准备最终结果
    final_metrics = [
        str(round(metrics['em']*100, 1)), 
        str(round(metrics["cover_em_1"]*100, 1)), 
        str(round(metrics['f1']*100, 1)),
        str(metrics["acc_num"]),
        str(round(metrics["cover_em_2"]*100, 1))
    ]
    
    if use_llm_judge:
        final_metrics.append(str(round(metrics["llm_judge"]*100, 1)))

    print("Eval File: ", file)
    print("EM: ", final_metrics[0])
    print("Cover-EM: ", final_metrics[1])
    print("Cover-EM_2: ", final_metrics[4])
    print("F1: ", final_metrics[2])
    print("Acc_Num: ", final_metrics[3])
    if use_llm_judge:
        print("LLM-Judge: ", final_metrics[5])

    overall_results = {
        'EM': final_metrics[0],
        'Cover-EM': final_metrics[1],
        'Cover-EM_2': final_metrics[4],
        'F1': final_metrics[2],
        'Acc_Num': final_metrics[3],
    }
    
    if use_llm_judge:
        overall_results['LLM-Judge'] = final_metrics[5]

    final_metrics_dict = {'overall': overall_results}
    output_file = file.replace(".jsonl", ".metrics.json")
    with open(output_file, mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics_dict, json_file, indent=4, ensure_ascii=False)
    
    return overall_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metric_eval.py <file_path> [--llm-judge] [--batch-llm]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    use_llm_judge = "--llm-judge" in sys.argv
    batch_llm_eval = "--batch-llm" in sys.argv
    
    result = eval(file_path, use_llm_judge=use_llm_judge, batch_llm_eval=batch_llm_eval)
    print("Final Results:", result)

# 使用示例:
# python metric_eval.py /path/to/file.jsonl  # 不使用LLM judge
# python metric_eval.py /path/to/file.jsonl --llm-judge  # 使用LLM judge (单个评估)
# python metric_eval.py /path/to/file.jsonl --llm-judge --batch-llm  # 使用LLM judge (批量评估)