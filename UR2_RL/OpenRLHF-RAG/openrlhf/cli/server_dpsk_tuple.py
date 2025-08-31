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
# from symeval import EvaluatorMathBatch

logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    ans = s.split("boxed")
    if len(ans) == 1:
        return s
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    return a


def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()


class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = datasets.load_from_disk(args.data_path).to_list()
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        print(len(self.eval_data_dict))
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

    def get_answer_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["answer"]
        return eval_data_dict

    def get_qa(self, query):

        question = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
        question = question.replace(
            "Please reason step by step, and put your final answer within \\boxed{}", ""
        ).strip()
        solution = query.split("<｜Assistant｜>")[-1].strip()
        return question, solution

    def get_query_answer(self, query):
        query = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
        query = query.replace("Please reason step by step, and put your final answer within \\boxed{}", "").strip()
        query = normalize_text(query)
        # print(query)
        # return self.eval_data_dict.get(query, "")
        return self.eval_data_dict[query]

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_thought(self, solution):
        thought = solution.split("<think>")[-1].strip().split("</think>")[0].strip()
        return thought

    def get_reward(self, queries):
        preds = []
        answers = []
        questions = []
        solutions = []
        finished_lst = []
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            question, solution = self.get_qa(queries[i])
            preds.append(self.get_query_pred(solution))
            answers.append(self.get_query_answer(question))
            questions.append(question)
            solutions.append(solution)
        logger.info(f"queries[0]: {queries[0]}")
        # print(preds, answers)

        # evaluator = EvaluatorMathBatch()
        # scores = evaluator.batch_eq(ref_answers=answers, pred_answers=preds)
        scores = [1] * len(queries)

        length_scores = []
        pattern_scores = []
        for i, query in enumerate(queries):
            self.cnt = self.cnt + 1
            if "boxed" not in solutions[i]:
                scores[i] = random.choice([-1.0, 0.0, 1.0, 2.0])
                finished_lst.append("0")
            else:
                if not scores[i]:
                    scores[i] = random.choice([-1.0, 0.0, 1.0, 2.0])
                    finished_lst.append("1")
                else:
                    scores[i] = random.choice([-1.0, 0.0, 1.0, 2.0])
                    finished_lst.append("1")

            if "boxed" not in query:
                length_scores.append(0)
            else:
                length_scores.append(0)

        # Write query-score pairs to JSONL if log_file is provided
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, f_f in zip(
                    questions,
                    solutions,
                    scores,
                    finished_lst,
                ):
                    record = {
                        "question": q,
                        "solution": a,
                        "score": s,
                        "finished": f_f,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # return scores
        assert len(scores) == len(length_scores)
        final_score = [[s0, s1] for s0, s1 in zip(scores, length_scores)]
        return final_score
        # return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSONL log file")

    args = parser.parse_args()

    # server
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        # print("sht-debug-"*30)
        # print(data)
        # print("sht-debug-"*30)
        queries = data.get("query")
        # print(queries)
        # print("sht-debug-"*30)
        # kill
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# python /home/songhuatong/OpenRLHF/openrlhf/cli/server_dpsk_tuple.py --data_path /home/songhuatong/OpenRLHF/data/still_dataset --reward_pretrain /home/songhuatong/Qwen2.5-1.5B-Instruct --log_file /home/songhuatong/sampling.jsonl --port 1278 --host 127.0.0.1