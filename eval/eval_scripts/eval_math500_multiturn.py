
import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from openai import OpenAI
import sys
import os
import re
from datasets import load_dataset
import json
import copy
from tqdm import tqdm
import requests
from collections import defaultdict
import time
from time import sleep
import multiprocessing
import torch
import random
from evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--src_file", type=str, default="/AIRPFS/lwt/eval/dataset/math500/test.jsonl")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="5004")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=3072)
    return parser.parse_args()

def process_text(example,model_short_name, tokenizer):
    if model_short_name == "qwen-2.5-7b-instruct":
        instruction = f"""
You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:

- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to 7.

Once you have all the information you need, continue your reasoning.

At the end of your reasoning, provide your final answer in the format \\boxed{{YOUR_ANSWER}}.

Example:
Question: "How do you compute the integral of e^(x^2) dx?"
Assistant thinking steps:
- I might need to look up techniques for integrating e^(x^ 2).

Assistant:
<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>

(System returns processed information from relevant web pages)

Assistant continues reasoning with the new information...

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.
- Provide the final answer in the format \\boxed{{YOUR_ANSWER}}.
"""     
        question = example["question"]
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n')
        messages_chat = [
            {"role": "user", "content": instruction + user_prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        example["chat_prompt"] = prompt
 
        return example

    base_prompt_r1_search = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""
    base_prompt_search_r1 ="""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: {question}\n"""
    base_prompt_r1_search_0_1_3 ="""---
The User asks a question, which belongs to either a **math** or a **retrieval (RAG)** task.  
The Assistant is required to solve it step by step, thinking carefully before answering.

The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly.  
This reasoning should be enclosed within `<think>` and `</think>` tags.  
Then, it provides the **final concise answer** within `<answer>` and `</answer>` tags.

If the question is about **mathematics**, the Assistant is strictly **forbidden to use search**, and must rely solely on its internal reasoning ability.

If the question is about **retrieval (RAG)**, the Assistant **should search if unsure** during the thinking phase using the following format:

<|begin\_of\_query|> keyword\_1 keyword\_2 keyword\_3 <|end\_of\_query|>

A search query must involve **only a single triple**.

The retrieved results will then be provided within:

<|begin\_of\_documents|> ...search results... <|end\_of\_documents|>

Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.
---

Output Format:

<think>
Step-by-step reasoning process goes here. Explain logic clearly and slowly.
Use search query only if necessary and allowed (i.e., for RAG only).
</think>

<answer>
Final concise answer goes here.
</answer>

User:{question}
Assistant: <think>"""
    base_prompt_r1_search_0_1_4 = """The User asks a question, and the Assistant solves it.
    The Assistant is required to solve it step by step, thinking carefully before answering.
    The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
    During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
    Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".
    The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.  The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

    User:{question}
    Assistant: <think>"""
    prompt_r1_ours_v1_1_0 = """You are sovling a math problem. Think step by step to solve it. 

The reasoning process includs detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""

## prompt of ur2 here
    prompt_r1_ours_v3_0_0 = """You are solving a math problem. Think step by step to solve it.

The reasoning process includes detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.

During your reasoning, if you're unsure about a factual concept — such as a definition, formula, theorem, or mathematical constant — you may issue a **search query** to clarify it.

Format your query using the following template (each query must target only one fact):

<|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

✅ Examples:
<|begin_of_query|> Definition of Möbius function <|end_of_query|>  
<|begin_of_query|> Formula for variance of Bernoulli distribution <|end_of_query|>

❌ Do NOT query for reasoning-related content like:
- Whether a solution approach is valid
- How to compute a specific value
- Multi-step deductions or conclusions

You may issue at most **four** search queries per problem — use them wisely.

When documents are returned in this format:
<|begin_of_documents|>  
... (search results here)  
<|end_of_documents|>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the question until you're confident in the answer.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
    if model_short_name == "qwen-2.5-7b-base-rl-ours-v0.1.3":
        question = example["question"]
        prompt = base_prompt_r1_search_0_1_3.format(question=question)
        example["chat_prompt"] = prompt    
    elif model_short_name == "qwen-2.5-7b-base-rl-ours-v0.1.4" or model_short_name == "qwen-2.5-7b-base-rl-ours-v0.1.5" or model_short_name == "qwen-2.5-7b-base-rl-ours-v0.1.6" or model_short_name == "qwen-2.5-7b-base-rl-ours-v0.1.7":
        question = example["question"]
        prompt = base_prompt_r1_search_0_1_4.format(question=question)
        example["chat_prompt"] = prompt    
    elif model_short_name == "qwen-2.5-7b-r1-search" or "qwen-2.5-7b-base-rl-ours-v0" in model_short_name:
        question = example["question"]
        prompt = base_prompt_r1_search.format(question=question)
        example["chat_prompt"] = prompt
    elif model_short_name == "qwen-2.5-7b-search-r1" or model_short_name == "qwen-2.5-7b-search-r1-it":
        question = example["question"]
        prompt = base_prompt_search_r1.format(question=question)
        example["chat_prompt"] = prompt
    elif "qwen-2.5-7b-instruct-rl-ours-v1" in model_short_name or model_short_name == "llama-3.1-8b-instruct-rl-ours-math-v0.6.0":
        question = example["question"]
        Question = f"Question: {question}"
        messages_chat=[
            {"role": "system", "content": prompt_r1_ours_v1_1_0},
            {"role": "user", "content": Question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        example["chat_prompt"] = prompt
    elif "qwen-2.5-7b-instruct-rl-ours-v3" in model_short_name or model_short_name == "llama-3.1-8b-instruct-rl-ours-v3.0.8-stage2" or model_short_name == "qwen-2.5-7b-instruct-rl-ours-v3.0.8-stage2":
        question = example["question"]
        Question = f"Question: {question}"
        messages_chat=[
            {"role": "system", "content": prompt_r1_ours_v3_0_0},
            {"role": "user", "content": Question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        example["chat_prompt"] = prompt
    elif "qwen-2.5-3b-instruct-rl-ours-v3" in model_short_name:
        question = example["question"]
        Question = f"Question: {question}"
        messages_chat=[
            {"role": "system", "content": prompt_r1_ours_v3_0_0},
            {"role": "user", "content": Question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        example["chat_prompt"] = prompt
    else:
        question = example["question"]
        Question = f"Question: {question}"
        messages_chat=[
            {"role": "system", "content": prompt_r1_ours_v1_1_0},
            {"role": "user", "content": Question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        example["chat_prompt"] = prompt        
    return example

SEARCH_FORMATS = {
    "qwen-2.5-7b-r1-search": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },
    "qwen-2.5-7b-base-rl-ours-v0.0": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },
    "qwen-2.5-7b-base-rl-ours-v0.1": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },
    "qwen-2.5-7b-base-rl-ours-v0.1.1": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },
    "qwen-2.5-7b-base-rl-ours-v0.1.2": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },  
    "qwen-2.5-7b-base-rl-ours-v0.1.3": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },   
    "qwen-2.5-7b-base-rl-ours-v0.1.4": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },  
    "qwen-2.5-7b-base-rl-ours-v0.1.5": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },  
    "qwen-2.5-7b-base-rl-ours-v0.1.6": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },  
    "qwen-2.5-7b-search-r1": {
        "begin_token": "<search>",
        "end_token": "</search>",
        "doc_begin": "<information>",
        "doc_end": "</information>",
    },
    "qwen-2.5-7b-search-r1-it": {
        "begin_token": "<search>",
        "end_token": "</search>",
        "doc_begin": "<information>",
        "doc_end": "</information>",
    },
     "qwen-2.5-7b-base-rl-ours-v0.1.7": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },    
    ## search token of ur2 here
     "qwen-2.5-7b-instruct-rl-ours-v1.1.0": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",
    },    
    "qwen-2.5-7b-instruct": {
        "begin_token": "<|begin_search_query|>",
        "end_token": "<|end_search_query|>",
        "doc_begin": "<|begin_search_result|>",
        "doc_end": "<|end_search_result|>",
    },
    
}

import re

def extract_answer_math(s):
    """
    从数学问题的输出中提取答案
    支持多种格式：\boxed{}, <answer></answer>, "the correct answer is"
    """
    extracted_text = ''
    
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
    
    # 所有匹配的结果
    all_matches = []
    
    # 1. 提取 \boxed{} 内容（使用平衡括号匹配）
    boxed_matches = extract_boxed_content(s)
    all_matches.extend(boxed_matches)
    
    # 2. 提取 <answer></answer> 内容
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, s, re.DOTALL | re.IGNORECASE)
    all_matches.extend(answer_matches)
    
    # 3. 提取 "the correct answer is" 后面的选项
    choice_pattern = r'[Tt]he correct answer is:?\s*([A-Ja-j])'
    choice_matches = re.findall(choice_pattern, s, re.IGNORECASE)
    all_matches.extend(choice_matches)
    
    # 返回最后一个匹配的结果（通常是最终答案）
    if all_matches:
        extracted_text = all_matches[-1]
    
    return extracted_text.strip()

def match_search_format(generated_text, stop_reason, model_short_name):
    if model_short_name not in SEARCH_FORMATS:
        model_short_name = "qwen-2.5-7b-instruct-rl-ours-v1.1.0"
    format_info = SEARCH_FORMATS[model_short_name]
    begin, end = format_info["begin_token"], format_info["end_token"]
    if begin in generated_text and stop_reason == end:
        return begin, end
    return None

def get_doc_insertion_text(model_short_name, doc_content):
    if model_short_name not in SEARCH_FORMATS:
        model_short_name = "qwen-2.5-7b-instruct-rl-ours-v1.1.0"
    format_info = SEARCH_FORMATS[model_short_name]
    return (
        f"{format_info['end_token']}\n\n"
        f"{format_info['doc_begin']}\n{doc_content}\n{format_info['doc_end']}\n\n"
    )


def main():
    print("=Begin="*10)
    args = parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    temp=args.temp
    port=args.port
    t_start = time.time()
    model_path=args.model_path
    model_short_name = model_path.split('/')[-1].lower()
    gpu_memory_rate=args.gpu_memory_rate
    data_ori_all = []
    result_path =""
    with open(args.src_file, "r") as f:
        data_ori_all = []
        for i, line in enumerate(f):
            if args.start_sample <= i < args.end_sample:
                obj_ori=json.loads(line)
                data_ori_all.append(obj_ori)
            if i >= args.end_sample - 1:
                break

    print("All Data Length: ",len(data_ori_all))
    chunk_size = 20000
    chunk_num = len(data_ori_all) // chunk_size
    if len(data_ori_all) % chunk_size != 0:
        chunk_num += 1
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=gpu_memory_rate, trust_remote_code=True)
    for h in range(chunk_num):
        print("==" * 80)
        print("Begin Chunk: ", h, "All: ", chunk_num)
        data_ori = data_ori_all[h * chunk_size:(h + 1) * chunk_size]
        data = [copy.deepcopy(item) for item in data_ori]
        # 处理 prompt
        for item in data:
            item = process_text(item, model_short_name, tokenizer)
            item["gen_text_store"] = ""

        finished_all_list = []
        if model_short_name == "qwen-2.5-7b-instruct":
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_search_query|>"]
        elif model_short_name == "qwen-2.5-7b-r1-search" or "qwen-2.5-7b-base-rl-ours-v0" in model_short_name:
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
        elif model_short_name == "llama-3.1-8b-instruct-rl-ours-math-v0.6.0":
            stop_tokens = ["<|eot_id|>", "<|endoftext|>"]    
        elif model_short_name == "llama-3.1-8b-instruct-rl-ours-v3.0.8-stage2":
            stop_tokens = ["<|eot_id|>", "<|endoftext|>", "<|end_of_query|>"]            
        elif "qwen-2.5-7b-search-r1" in model_short_name:
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>", "</answer>"]
        else:
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
        sampling_params = SamplingParams(temperature=temp, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens, stop=stop_tokens)

        finished_all_list=[]
        
        continued_answer = copy.deepcopy(data)
        for k in range(16):
            prompts = [item["chat_prompt"] for item in continued_answer]
            outputs = llm.generate(prompts, sampling_params)
            finished_texts = []
            continued_texts = []

            gen_finished_texts = []
            query_list=[]
            prev_reasonings = []
            for i, output in enumerate(outputs):

                prompt = output.prompt
                answer = continued_answer[i]["answer"]
                quesiton = continued_answer[i]["question"]
                gen_text_store = continued_answer[i]["gen_text_store"]
                stop_reason = output.outputs[0].stop_reason
                generated_text = output.outputs[0].text
                if k == 5: #检索次数太多了，直接停掉，就是未完成
                    original_data = {
                            "question":quesiton,
                            "answer": answer,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "generated_text":generated_text,
                            "stop_reason_final": "many_retrieve",
                            "pred_ans": "I don't know."
                        }

                    finished_texts.append(original_data)
                    continue
                matched = match_search_format(generated_text, stop_reason, model_short_name)         
                if extract_answer_math(generated_text) != '':
                    pred_ans = extract_answer_math(generated_text)
                    original_data = {
                        "question":quesiton,
                        "answer": answer,
                        "pred_ans": pred_ans,
                        "stop_reason_final": "finished",
                        "gen_text_store": gen_text_store + generated_text,
                    }   
                    finished_texts.append(original_data)
                elif matched:
                    begin_token, end_token = matched
                    query = generated_text.split(begin_token)[-1].split(end_token)[0]
                    query = query.replace('"',"").replace("'","").replace("\t"," ").replace("...","")
                    prev_reasonings.append(gen_text_store + generated_text.split(begin_token)[0])
                    if query:
                        topk = args.topk
                        gen_finished_texts.append(None)
                        query_list.append(query)

                        original_data = {
                            "chat_prompt": prompt + generated_text.strip(),
                            "answer": answer,
                            "question": quesiton,
                            "stop_reason": stop_reason,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                        }
                        continued_texts.append(original_data)
                    else:
                        original_data = {
                            "question": quesiton,
                            "answer": answer,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "generated_text": generated_text,
                            "stop_reason_final": "query_inst_error",
                            "pred_ans": "I don't know."
                        }
                        finished_texts.append(original_data)
                else:
                    original_data = {
                    "question":quesiton,
                    "answer": answer,
                    "gen_text_store": gen_text_store + generated_text.strip(),
                    "stop_reason_final": "shot_down",
                    "pred_ans": "I don't know."
                }
                    finished_texts.append(original_data)

            print("=="*80)

            assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
            url_wiki = "http://localhost:"+port+"/queries"
            if len(query_list)!=0:
                status_code = 0
                while status_code!=200:
                    response = requests.post(url_wiki, json={"queries": query_list, "prev_reasonings": prev_reasonings, "k": topk, "sources":["math" for _ in range(len(query_list))]})
                    status_code = response.status_code
                    value = random.uniform(0.01, 0.5)
                    time.sleep(value)
                if response.status_code == 200:
                    result = response.json()
                    answers = result["answers"]
                    for i in range(len(answers)):
                        retrieve_docs = answers[i]
                        continued_text_now = copy.deepcopy(continued_texts[i])
                        if len(retrieve_docs)>0:
                            doc_content_list = []
                            for j in range(len(retrieve_docs)):
                                doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                doc_content_list.append(f"{doc_now.strip()}\n")
                            doc_content = ''.join(doc_content_list)

                        else:
                            doc_content = "None"
                        doc_insertion = get_doc_insertion_text(model_short_name, doc_content)
                        continued_text_now["chat_prompt"] += doc_insertion
                        continued_text_now["gen_text_store"] += doc_insertion
                        continued_texts[i] = continued_text_now
                else:
                    for i in range(len(continued_texts)):
                        current_data = continued_texts[i]  # 临时保存引用
                        original_data = {
                            "question": current_data["question"],
                            "answer": current_data["answer"],
                            "stop_reason_final": "retrieve_error",
                        }
                        continued_texts[i] = copy.deepcopy(original_data)
                    # raise Exception("Error in response: the status code is not 200!")

            finished_all_list.extend(finished_texts)
            print("=="*80)
            print("Epoch: ",k,"New_Finished: ",len(finished_texts),"All_Finished ",len(finished_all_list),"Continued: ",len(continued_texts))

            print("Begin Writing Epoch: ",k)

            # print(continued_texts)
            print("=="*80)
            # print(finished_texts)
            if len(continued_texts)!=0:
                continued_answer = copy.deepcopy(continued_texts)
            else:
                continued_answer = []
                break            
    input_list = []
    output_list = []

    for item in finished_all_list:
        input_prompt = process_text(item,model_short_name, tokenizer)['chat_prompt']
        input_list.append(input_prompt)
        output_list.append(item['gen_text_store'])
    if "math500" in args.src_file:
        output_dir = f'../outputs/math500/{model_short_name}.rag'
    elif "amc23.jsonl" in args.src_file:
        output_dir = f'../outputs/amc23/{model_short_name}.rag'
    elif "amc.jsonl" in args.src_file:
        output_dir = f'../outputs/amc/{model_short_name}.rag'
    elif "minervamath" in args.src_file:
        output_dir = f'../outputs/minervamath/{model_short_name}.rag'
    elif "olympiad" in args.src_file:
        output_dir = f'../outputs/olympiad/{model_short_name}.rag'
    os.makedirs(output_dir, exist_ok=True)
    split = "test"
    run_evaluation(        
        finished_all_list,
        input_list,
        output_list,
        output_dir,
        time.time()-t_start,
        split,
    )
    if dist.is_initialized():
            dist.destroy_process_group()
if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()