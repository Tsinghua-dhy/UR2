import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets
import random
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer
import sys
import string
from collections import Counter
import pickle
from math_equivalence import is_equiv
import asyncio
from judge_math_answer_gpt import eval_math_scores
logger = init_logger(__name__)

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

def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )

def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    return all(ground in pre_list for ground in ground_list)

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

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text

def extract_answer_math(s):
    extracted_text = ''
    patterns = [
        r'\\boxed\{(.*)\}',  # 匹配 LaTeX 格式的 \boxed{...}
        r'<answer>(.*?)</answer>',  # 匹配 <answer>...</answer>
        r'[Tt]he correct answer is:?\s*([A-Ja-j])'  # 匹配 the/The correct answer is[:] A-J/a-j
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, s, re.DOTALL | re.IGNORECASE))
    if matches:
        extracted_text = matches[-1]  # 取最后一个匹配
    return extracted_text.strip()

def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：""《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = []
        with open(args.data_path, 'r') as f:
            for line in f:
                eval_dataset.append(json.loads(line))
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        self.eval_source_dict = self.get_source_dict(eval_dataset)
        self.eval_idx_dict = self.get_idx_dict(eval_dataset)
        self.eval_level_dict = self.get_level_dict(eval_dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True, use_fast=True)
        self.log_file = args.log_file
        self.avg_length_dict = []
        self.cnt = 0
        self.avg_len = 5000
        self.key_words = [
            "wait",
            "double check",
            "what",
            "how",
            "why",
            "alternatively",
            "think",
            "rethink",
            "?",
            "change",
            "try",
            "check",
        ]

    def get_idx_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["idx"]
        return eval_data_dict

    def get_answer_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["answer"]
        return eval_data_dict

    def get_source_dict(self, eval_dataset):
        eval_source_dict = {}
        for item in eval_dataset:
            eval_source_dict[normalize_text(item["question"])] = item["source"]
        return eval_source_dict

    def get_level_dict(self, eval_dataset):
        eval_level_dict = {}
        for item in eval_dataset:
            eval_level_dict[normalize_text(item["question"])] = item["level"]
        return eval_level_dict

    def get_qa(self, query):
        if "\nOptions:\nA. " in query:
            remove_prefix = " ".join(query.split("Question:")[1:])
            question = remove_prefix.split("Options:")[0].strip()
            solution = remove_prefix.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[-1].strip()
        else:
            question = query.split("<|start_header_id|>user<|end_header_id|>\n\n")[-1].split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[0].strip().split("Question: ")[-1].strip()
            solution = query.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[-1].strip()
        is_retrieve = "<|begin_of_query|>" in query
        return question, solution, is_retrieve

    def get_query_source(self, query):
        query = normalize_text(query)
        return self.eval_source_dict.get(query, 'unknown')

    def get_query_level(self, query):
        query = normalize_text(query)
        return self.eval_level_dict.get(query, 'unknown')

    def get_query_idx(self, query):
        query = normalize_text(query)
        return self.eval_idx_dict.get(query, 'unknown')

    def get_query_answer(self, query):
        query = normalize_text(query)
        return self.eval_data_dict.get(query, '')

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_reward(self, queries, epoch):
        preds = []
        answers = []
        questions = []
        solutions = []
        sources = []
        finished_lst = []
        idxs = []
        is_equivs = []
        levels = []
        is_retrieves = []
        scores = []
        format_rewards = []
        format_error_reasons = []  # 新增：记录格式错误原因

        for i in range(len(queries)):
            self.tokenizer.pad_token=self.tokenizer.eos_token
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            question, solution, is_retrieve = self.get_qa(queries[i])
            is_retrieves.append(is_retrieve)
            preds.append(self.get_query_pred(solution))
            answers.append(self.get_query_answer(question))
            sources.append(self.get_query_source(question))
            idxs.append(self.get_query_idx(question))
            levels.append(self.get_query_level(question))
            questions.append(question)
            solutions.append(solution)
            format_error_reasons.append([])  # 初始化错误原因列表
            # Check mathematical equivalence for math questions
            if sources[i] == 'math':
                is_equivs.append(is_equiv(answers[i], preds[i]))
            else:
                is_equivs.append(False)

        # Evaluate math scores using GPT-4o-mini for math questions
        
        math_scores = eval_math_scores(questions, answers, preds, sources, is_equivs)
        for i in range(len(queries)):
            if sources[i] == 'math':
                scores.append(math_scores[i])
            elif sources[i] == 'rag':
                f1, _, _ = f1_score(preds[i], answers[i])
                scores.append(2.0 * float(f1))
            else:
                score = 2.0 if preds[i].lower() == answers[i].lower() else 0.0
                scores.append(score)
            format_rewards.append(0.0)  # Initialize format_rewards
        for i, query in enumerate(queries):
            self.cnt = self.cnt + 1
            if preds[i] == '':
                scores[i] = 0.0
                finished_lst.append("0")
            else:
                finished_lst.append("1")

            # Skip format checks for fallback cases
            if not questions[i] or normalize_text(questions[i]) not in self.eval_data_dict:
                continue

            format_punishment = False
            retrieve_reward = 0.0
            is_retrieve = is_retrieves[i]

            # 检查格式问题并记录原因
            count_answer = len(re.findall(r'[Tt]he correct answer is:?\s*([A-Ja-j])', solutions[i], re.IGNORECASE)) + len(re.findall(r'\\boxed\{(.*)\}', solutions[i], re.IGNORECASE))
            if count_answer != 1:
                format_punishment = True
                if count_answer == 0:
                    format_error_reasons[i].append("No answer format found (missing boxed or 'the correct answer is')")
                else:
                    format_error_reasons[i].append(f"Multiple answer formats found ({count_answer} instances)")
                    
            count_1 = solutions[i].count("<|begin_of_query|>")
            count_2 = solutions[i].count("<|end_of_query|>")
            count_3 = solutions[i].count("<|begin_of_documents|>")
            count_4 = solutions[i].count("<|end_of_documents|>")
            count_5 = solutions[i].count("This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning.") 
            if count_1 != count_2 or count_3 != count_4 or count_1 != count_3 or count_1 > 4:
                format_punishment = True
                format_error_reasons[i].append(f"Query/document tags mismatch (begin_query:{count_1}, end_query:{count_2}, begin_doc:{count_3}, end_doc:{count_4})")
                
            tool_calls = re.findall(r'<\|begin_of_query\|>\s*(.*?)\s*<\|end_of_query\|>', solutions[i], re.DOTALL)
            for call in tool_calls:
                if len(call.split()) > 20:
                    format_punishment = True
                    format_error_reasons[i].append(f"Tool call too long ({len(call.split())} words, max 20)")
                    
            special_tokens = ['<|start_header_id|>user<|end_header_id|>', '<|start_header_id|>assistant<|end_header_id|>', '<|start_header_id|>', '<|end_header_id|>']
            for special_token in special_tokens:
                if special_token in solutions[i]:
                    format_punishment = True
                    format_error_reasons[i].append(f"Contains special token: {special_token}")
                    
            have_chinese = any('\u4e00' <= char <= '\u9fff' for char in solutions[i])
            if have_chinese:
                format_punishment = True
                format_error_reasons[i].append("Contains Chinese characters")
                
            answer_len = len(preds[i].split())
            if sources[i] not in ['rag', 'math']:
                if answer_len > 1 or answer_len == 0:
                    format_punishment = True
                    format_error_reasons[i].append(f"Invalid answer length for non-rag/math question ({answer_len} words)")
            else:
                if answer_len > 10 and answer_len > len(answers[i].split()):
                    format_punishment = True
                    format_error_reasons[i].append(f"Answer too long ({answer_len} words, expected max {len(answers[i].split())} or 10)")
                    
            if "<|begin_of_query|>" in preds[i] or "<|begin_of_documents|>" in preds[i]:
                format_punishment = True
                format_error_reasons[i].append("Answer contains query/document tags")
                
            if is_retrieve == False:
                if len(tool_calls) > 0:
                    format_punishment = True
                    format_error_reasons[i].append("Non-retrieve question contains tool calls")
                if format_punishment != True:
                    scores[i] = scores[i] + 1.0
                    format_rewards[i] = format_rewards[i] + 1.0
            else:
                if count_1 == 0:
                    format_punishment = True
                    format_error_reasons[i].append("Missing query/document tags (<|begin_of_query|> or <|begin_of_documents|>)")
                if count_1 == 1:
                    retrieve_reward = 0.5
                elif count_1 >= 2:medqa_mmlu
                    retrieve_reward = 1.0

                if count_5 > 0:
                    format_rewards[i] -= 0.5*count_5
                    scores[i] -= 0.5*count_5
                    format_error_reasons[i].append("Contains fallback message for complex queries")

                if format_punishment != True:
                    scores[i] = scores[i] + 1.0
                    format_rewards[i] = format_rewards[i] + 1.0
                    
                if retrieve_reward > 0 and epoch < 10:
                    scores[i] = scores[i] + retrieve_reward
                    format_rewards[i] = format_rewards[i] + retrieve_reward

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, f_f, idx, level, source, pred, ans, format, is_retrieve, error_reasons in zip(
                    questions,
                    solutions,
                    scores,
                    finished_lst,
                    idxs,
                    levels,
                    sources,
                    preds,
                    answers,
                    format_rewards,
                    is_retrieves,
                    format_error_reasons
                ):
                    record = {
                        "question": q,
                        "solution": a,
                        "score": s,
                        "idx": idx,
                        "finished": f_f,
                        "epoch": epoch,
                        "level": level,
                        "type": source,
                        "pred": pred,
                        "ans": ans,
                        "format_reward": format,
                        "is_retrieve": is_retrieve,
                        "format_error_reasons": error_reasons if error_reasons else None  # 新增字段
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return scores, format_rewards, sources, is_retrieves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSONL log file")

    args = parser.parse_args()
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        epoch = data.get("epoch")
        rewards = reward_model.get_reward(queries, epoch)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")