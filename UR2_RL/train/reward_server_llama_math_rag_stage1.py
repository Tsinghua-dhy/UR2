import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from openrlhf.utils.logging_utils import init_logger
import string

logger = init_logger(__name__)

def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：""《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

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

class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = []
        with open(args.data_path, 'r') as f:
            for line in f:
                eval_dataset.append(json.loads(line))
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        self.eval_source_dict = self.get_source_dict(eval_dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True, use_fast=True)
        self.log_file = args.log_file
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

    def get_query_answer(self, query):
        query = normalize_text(query)
        return self.eval_data_dict.get(query, '')

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_reward(self, queries, epoch):
        questions = []
        solutions = []
        sources = []
        is_retrieves = []
        format_rewards = []
        format_error_reasons = []

        for i in range(len(queries)):
            self.tokenizer.pad_token=self.tokenizer.eos_token
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            question, solution, is_retrieve = self.get_qa(queries[i])
            is_retrieves.append(is_retrieve)
            questions.append(question)
            solutions.append(solution)
            sources.append(self.get_query_source(question))
            format_error_reasons.append([])  # 初始化错误原因列表
            format_rewards.append(0.0)  # 初始化 format_rewards

        for i, query in enumerate(queries):
            format_punishment = 0
            retrieve_reward = 0.0
            is_retrieve = is_retrieves[i]
            pred = self.get_query_pred(solutions[i])
            answer = self.get_query_answer(questions[i])

            # 检查格式问题并记录原因
            count_answer = len(re.findall(r'[Tt]he correct answer is:?\s*([A-Ja-j])', solutions[i], re.IGNORECASE)) + len(re.findall(r'\\boxed\{(.*)\}', solutions[i], re.IGNORECASE))
            if count_answer != 1:
                format_punishment += 1
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
                format_punishment += 1
                format_error_reasons[i].append(f"Query/document tags mismatch (begin_query:{count_1}, end_query:{count_2}, begin_doc:{count_3}, end_doc:{count_4})")

            tool_calls = re.findall(r'<\|begin_of_query\|>\s*(.*?)\s*<\|end_of_query\|>', solutions[i], re.DOTALL)
            for call in tool_calls:
                if len(call.split()) > 20:
                    format_punishment += 1
                    format_error_reasons[i].append(f"Tool call too long ({len(call.split())} words, max 20)")

            special_tokens = ['<|start_header_id|>user<|end_header_id|>', '<|start_header_id|>assistant<|end_header_id|>', '<|start_header_id|>', '<|end_header_id|>']
            for special_token in special_tokens:
                if special_token in solutions[i]:
                    format_punishment += 1
                    format_error_reasons[i].append(f"Contains special token: {special_token}")

            have_chinese = any('\u4e00' <= char <= '\u9fff' for char in solutions[i])
            if have_chinese:
                format_punishment += 1
                format_error_reasons[i].append("Contains Chinese characters")

            answer_len = len(pred.split())
            if sources[i] not in ['rag', 'math']:
                if answer_len > 1 or answer_len == 0:
                    format_punishment += 1
                    format_error_reasons[i].append(f"Invalid answer length for non-rag/math question ({answer_len} words)")
            else:
                if answer_len > 10 and answer_len > len(answer.split()):
                    format_punishment += 1
                    format_error_reasons[i].append(f"Answer too long ({answer_len} words, expected max {len(answer.split())} or 10)")

            if "<|begin_of_query|>" in pred or "<|begin_of_documents|>" in pred:
                format_punishment += 1
                format_error_reasons[i].append("Answer contains query/document tags")

            if is_retrieve == False:
                if len(tool_calls) > 0:
                    format_punishment += 1
                    format_error_reasons[i].append("Non-retrieve question contains tool calls")
                if not format_punishment:
                    format_rewards[i] = format_rewards[i] + 1.0
            else:
                if count_1 == 0 and epoch < 15:
                    format_punishment += 1
                    format_error_reasons[i].append("Missing query/document tags (<|begin_of_query|> or <|begin_of_documents|>)")
                elif count_1 >= 1:
                    retrieve_reward = 3.0

                if count_5 > 0:
                    format_rewards[i] -= 0.5 * count_5
                    format_error_reasons[i].append("Contains fallback message for complex queries")

                if format_punishment == 0:
                    format_rewards[i] = format_rewards[i] + 1.0
                else:
                    format_rewards[i] -= format_punishment
                if retrieve_reward > 0 and epoch < 15:
                    format_rewards[i] = format_rewards[i] + retrieve_reward

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, format, is_retrieve, error_reasons in zip(
                    questions,
                    solutions,
                    sources,
                    format_rewards,
                    is_retrieves,
                    format_error_reasons
                ):
                    record = {
                        "question": q,
                        "solution": a,
                        "type": s,
                        "format_reward": format,
                        "is_retrieve": is_retrieve,
                        "format_error_reasons": error_reasons if error_reasons else None
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return format_rewards, format_rewards, sources, is_retrieves

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