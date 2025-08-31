import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from utils.math_equivalence import is_equiv
from utils.judge_math_answer_gpt import eval_gpt


def extract_answer(output, mode='gen'):
    # 方法1：使用平衡括号匹配处理 \boxed{}
    def extract_boxed_content(text):
        """提取 \boxed{} 中的内容，正确处理嵌套括号"""
        pattern = r'\\boxed\{'
        matches = []
        
        for match in re.finditer(pattern, text):
            start_pos = match.end() - 1  # 定位到开始的 '{'
            brace_count = 0
            content_start = start_pos + 1
            
            for i in range(start_pos, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到匹配的结束括号
                        matches.append(text[content_start:i])
                        break
        
        return matches
    """
    从输出中提取答案
    
    Args:
        output: 包含答案的文本
        mode: 'gen' 或 'rat' 模式
    
    Returns:
        提取的答案文本
    """
    extracted_text = ''
    
    # 预处理：如果没有"So the final answer is"，则尝试分割
    if "So the final answer is" not in output:
        output = output.split("</think>")[-1].split("Final Answer")[-1].strip()
    
    if mode == 'gen':
        # 支持匹配 \boxed{...} 和 <answer>...</answer>
        """
        从数学问题的输出中提取答案
        支持多种格式：\boxed{}, <answer></answer>, "the correct answer is"
        """
        extracted_text = ''

        # 所有匹配的结果
        all_matches = []
        
        # 1. 提取 \boxed{} 内容（使用平衡括号匹配）
        boxed_matches = extract_boxed_content(output)
        all_matches.extend(boxed_matches)
        
        # 2. 提取 <answer></answer> 内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, output, re.DOTALL | re.IGNORECASE)
        all_matches.extend(answer_matches)
        
        # 3. 提取 "the correct answer is" 后面的选项
        choice_pattern = r'[Tt]he correct answer is:?\s*([A-Ja-j])'
        choice_matches = re.findall(choice_pattern, output, re.IGNORECASE)
        all_matches.extend(choice_matches)
        
        # 返回最后一个匹配的结果（通常是最终答案）
        if all_matches:
            extracted_text = all_matches[-1]
        
        return extracted_text.strip()
            
    elif mode == 'rat':
        # 使用正则表达式匹配最后一个 \boxed{...}
        # 这个正则会匹配嵌套的大括号
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, output)
        
        if matches:
            # 取最后一个匹配的内容
            extracted_text = matches[-1].strip()
        else:
            # 如果没找到 \boxed，尝试其他模式
            # 可以返回空字符串或尝试其他提取方式
            extracted_text = output
            
    return extracted_text.strip()

def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
  
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    normalized_pred_answer = normalize_answer(pred_answer)
    normalized_ground_truth = normalize_answer(labeled_answer)
    em = int(normalized_pred_answer == normalized_ground_truth)
    acc = int(normalized_ground_truth in normalized_pred_answer)

    prediction_tokens = normalized_pred_answer.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        f1 = 0
    else:
        precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
        recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

    final_metric["em"] = em
    final_metric["acc"] = acc
    final_metric["f1"] = f1
    final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    return final_metric, pred_answer

def run_evaluation(filtered_data, input_list, output_list, output_dir, total_time, split, mode="gen"):
    avg_em, avg_acc, avg_f1, avg_math, avg_math_gpt = [], [], [], [], []
    num_valid_answer = 0

    for item, input_prompt, result in zip(filtered_data, input_list, output_list):
        if type(result) == str:
            item['Output'] = result
        else:
            item['Output'] = result.outputs[0].text
        
        labeled_answer = item["answer"]    
        metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode)

        item['pred_ans'] = pred_answer
        item['Metrics'] = metric
        item['Question'] = input_prompt

        if pred_answer != '':
            num_valid_answer += 1

        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])

    filtered_data = eval_gpt(filtered_data)
    for item in filtered_data:
        avg_math_gpt.append(item['Metrics']['math_equal_gpt'])
    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

    overall_results = {
        'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
        'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
        'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
        'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
        'math_equal_gpt': np.mean(avg_math_gpt) if len(avg_em) > 0 else 0.0,
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
    }

    final_metrics = {'overall': overall_results}

    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)