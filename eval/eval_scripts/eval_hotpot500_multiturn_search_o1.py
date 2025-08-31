import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import os
import re
from datasets import load_dataset
import copy
from tqdm import tqdm
import requests
from collections import defaultdict
import time
import random
from metric_calc_rule import eval

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on HotpotQA dataset with multi-round retrieval.")
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--src_file", type=str, default="/AIRPFS/lwt/eval/dataset/hotpot500/hotpotqa_500.jsonl")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_rounds", type=int, default=10, help="Maximum number of retrieval rounds.")
    return parser.parse_args()

def get_cot_prompt(question, model_short_name, tokenizer,args):
    if model_short_name == "qwen-2.5-7b-instruct":
        sys_prompt = f"""
You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:

- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {args.max_rounds}.

Once you have all the information you need, continue your reasoning.

At the end of your reasoning, provide your final answer in the format \\boxed{{YOUR_ANSWER}}.

Example:
Question: "Alice David is the voice of Lara Croft in a video game developed by which company?"
Assistant thinking steps:
- I need to find out who voices Lara Croft in the video game.
- Then, I need to determine which company developed that video game.

Assistant:
<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>

(System returns processed information from relevant web pages)

Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.

Assistant:
<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>

(System returns processed information from relevant web pages)

Assistant continues reasoning with the new information...

Finally, the assistant provides the answer:
\\boxed{{Square Enix}}

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.
- Provide the final answer in the format \\boxed{{YOUR_ANSWER}}.
"""     
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
        messages_chat = [
            {"role": "user", "content": sys_prompt + user_prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        return prompt
    sys_prompt = """You are solving a factual open-domain question from a Knowledge Question Answering (KQA) task. The question requires step-by-step reasoning over real-world knowledge to identify a specific, factually correct answer.

Carefully analyze the question to understand the key entities, relationships, and constraints involved. Retrieve and consider relevant factual knowledge, and reason logically to identify the most accurate answer. The reasoning process should include detailed considerations such as analyzing the question, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps when needed.

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<search> your concise query (less than 20 words) </search>

* You can issue **multiple queries** at different steps in your reasoning.  
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

Once documents are returned in this format:  
<information>  
... (search results here)  
</information>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the question until you're confident in the answer.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
    Question = f"Question: {question}"
    messages_chat = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": Question}
    ]
    prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
    if model_short_name == "qwen-2.5-7b-search-r1" or model_short_name == "qwen-2.5-3b-search-r1":
        base_prompt_search_r1 = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: {question}\n"""
        prompt = base_prompt_search_r1.format(question=question)
    elif model_short_name == "qwen-2.5-7b-search-r1-it":
        base_prompt_search_r1 = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: {question}\n"""
        prompt = base_prompt_search_r1.format(question=question)
        messages_chat = [
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
    elif model_short_name == "llama-3.1-8b-instruct-r1-search":
        messages_chat=[
            {"role": "system","content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""},
            {"role": "user", "content":question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"
    elif model_short_name == "qwen-2.5-7b-instruct-rl-ours-v1.0.3":
        sys_prompt = """You are a helpful assistant.
    Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
    The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
    During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|44begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
    Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""
        Question = f"{question}"
        messages_chat = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": Question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"        
    elif "qwen-2.5-7b-instruct-rl-ours-v2.0.0" in model_short_name or "qwen-2.5-7b-instruct-rl-ours-v1.1." in model_short_name or "qwen-2.5-3b-instruct-rl-ours-v1.0.4" in model_short_name or "qwen-2.5-3b-instruct-rl-ours-v1.1" in model_short_name or "qwen-2.5-7b-instruct-rl-ours-v3.0" in model_short_name or  "qwen-2.5-3b-instruct-rl-ours-v3" in model_short_name or model_short_name == "qwen-2.5-7b-instruct-rl-ours-vablation-data" or model_short_name == "llama-3.1-8b-instruct-rl-ours-v0.1.4" or model_short_name == "llama-3.1-8b-instruct-rl-ours-v3.0.8-stage2" or model_short_name == "qwen-2.5-7b-instruct-rl-ours-v3.0.8-stage2":
## prompt of ur2 here 
        sys_prompt = """You are solving a factual open-domain question from a Knowledge Question Answering (KQA) task. The question requires step-by-step reasoning over real-world knowledge to identify a specific, factually correct answer.

Carefully analyze the question to understand the key entities, relationships, and constraints involved. Retrieve and consider relevant factual knowledge, and reason logically to identify the most accurate answer. 

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

* You can issue **multiple queries** at different steps in your reasoning.
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.
  * ✅ Example:
    * <|begin_of_query|> When did Einstein move to the United States? <|end_of_query|>
    * <|begin_of_query|> Why did Einstein leave Germany? <|end_of_query|>
  * ❌ Do **not** combine them like this:
    * <|begin_of_query|> When did Einstein move to the US and why did he leave Germany? <|end_of_query|>
* You may issue **at most five queries** in total — use them wisely.
Once documents are returned in this format:  
<|begin_of_documents|> 
... (search results here)  
<|end_of_documents|>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the question until you're confident in the answer.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
        Question = f"{question}"
        messages_chat=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": Question}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
    elif "instruct-zerosearch" in model_short_name:
        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
        messages_chat=[
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) 
    elif "qwen-2.5-7b-instruct-rl-ours-v" in model_short_name:
        prompt = prompt  # For instruct models, append <think> to align with expected format
    elif model_short_name == "qwen-2.5-7b-r1-search":
        base_prompt_r1_search = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""
        prompt = base_prompt_r1_search.format(question=question)
    else:
        base_prompt_r1_search_0_1_4 = """The User asks a question, and the Assistant solves it.
The Assistant is required to solve it step by step, thinking carefully before answering.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".
The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage. The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

User:{question}
Assistant: <think>"""
        prompt = base_prompt_r1_search_0_1_4.format(question=question)
    return prompt

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
    "qwen-2.5-7b-base-rl-ours-v0.1.7": {
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
    "qwen-2.5-3b-search-r1": {
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
    "qwen-2.5-7b-instruct-rl-ours-v1.0.3": {
        "begin_token": "<|begin_of_query|>",
        "end_token": "<|end_of_query|>",
        "doc_begin": "<|begin_of_documents|>",
        "doc_end": "<|end_of_documents|>",  
    },
    "qwen-2.5-7b-instruct-rl-ours-v1.0.0": {
        "begin_token": "<search>",
        "end_token": "</search>",
        "doc_begin": "<information>",
        "doc_end": "</information>",       
    },
    "qwen-2.5-7b-instruct-zerosearch": {
        "begin_token": "<search>",
        "end_token": "</search>",
        "doc_begin": "<information>",
        "doc_end": "</information>",
    },
    "qwen-2.5-3b-instruct-zerosearch": {
        "begin_token": "<search>",
        "end_token": "</search>",
        "doc_begin": "<information>",
        "doc_end": "</information>",
    },
    "qwen-2.5-7b-instruct": {
        "begin_token": "<|begin_search_query|>",
        "end_token": "<|end_search_query|>",
        "doc_begin": "<|begin_search_result|>",
        "doc_end": "<|end_search_result|>",
    }
}

def match_search_format(generated_text, stop_reason, model_short_name):
    if model_short_name not in SEARCH_FORMATS:
        model_short_name = "qwen-2.5-7b-instruct-rl-ours-v1.0.3"
    format_info = SEARCH_FORMATS[model_short_name]
    begin, end = format_info["begin_token"], format_info["end_token"]
    if begin in generated_text and stop_reason == end:
        return begin, end
    return None

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

def get_doc_insertion_text(model_short_name, doc_content):
    if model_short_name not in SEARCH_FORMATS:
        model_short_name = "qwen-2.5-7b-instruct-rl-ours-v1.0.3"
    format_info = SEARCH_FORMATS[model_short_name]
    return (
        f"{format_info['end_token']}\n\n"
        f"{format_info['doc_begin']}\n{doc_content}\n{format_info['doc_end']}\n\n"
    )

def main():
    print("=Begin="*10)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model_short_name = args.model_path.split('/')[-1].lower()
    
    # Load data
    data = []
    with open(args.src_file, "r") as f:
        for i, line in enumerate(f):
            if args.start_sample <= i < args.end_sample:
                data.append(json.loads(line))
            if i >= args.end_sample - 1:
                break
    print("All Data Length: ", len(data))

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_rate,
        trust_remote_code=True
    )

    # Prepare data
    continued_answer = []
    for item in data:
        question = item["question"]
        answer = item["answer"]
        prompt = get_cot_prompt(question, model_short_name, tokenizer, args)
        continued_answer.append({
            "chat_prompt": prompt,
            "question": question,
            "answer": answer,
            "gen_text_store": ""
        })

    # Set stop tokens
    if model_short_name == "qwen-2.5-7b-instruct":
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_search_query|>"]
    elif model_short_name == "qwen-2.5-7b-r1-search" or "qwen-2.5-7b-base-rl-ours-v0" in model_short_name or model_short_name == "qwen-2.5-7b-instruct-rl-ours-v1.0.3" or "qwen-2.5-7b-instruct-rl-ours-v1.1." in model_short_name:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
    elif model_short_name == "llama-3.1-8B-instruct-r1-search" or model_short_name == "llama-3.1-8b-instruct-rl-ours-v0.1.4" or model_short_name == "llama-3.1-8b-instruct-rl-ours-v3.0.8-stage2":
        stop_tokens = ["<|eot_id|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]        
    elif "qwen-2.5-7b-search-r1" in model_short_name or "qwen-2.5-3b-search-r1" in model_short_name:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>", "</answer>"]
    elif "instruct-zerosearch" in model_short_name:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>", "</answer>"]
    elif "qwen-2.5-7b-instruct-rl-ours-v1" in model_short_name:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>", "</answer>"]
    else:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=stop_tokens
    )
    print(model_short_name)
    # Multi-round inference and retrieval
    finished_all_list = []
    for k in range(args.max_rounds):
        prompts = [item["chat_prompt"] for item in continued_answer]
        outputs = llm.generate(prompts, sampling_params)
        finished_texts = []
        continued_texts = []
        query_list = []
        prev_reasonings = []
        for i, output in enumerate(outputs):
            prompt = output.prompt
            answer = continued_answer[i]["answer"]
            question = continued_answer[i]["question"]
            gen_text_store = continued_answer[i]["gen_text_store"]
            stop_reason = output.outputs[0].stop_reason
            generated_text = output.outputs[0].text

            if k == args.max_rounds - 1:  # Too many retrievals, stop
                original_data = {
                    "question": question,
                    "answer": answer,
                    "generated_text": generated_text,
                    "stop_reason_final": "many_retrieve",
                    "pred_ans": "I don't know."
                }
                finished_texts.append(original_data)
                continue

            matched = match_search_format(generated_text, stop_reason, model_short_name)
            if "<answer>" in generated_text and stop_reason == "</answer>" and (model_short_name == "qwen-2.5-7b-r1-search" or model_short_name == "qwen-2.5-7b-search-r1" or model_short_name == "qwen-2.5-7b-search-r1-it" or model_short_name == "qwen-2.5-7b-instruct-zerosearch" or model_short_name == "llama-3.1-8b-instruct-r1-search" or model_short_name == "qwen-2.5-3b-search-r1"):
                original_data = {
                    "question": question,
                    "answer": answer,
                    "pred_ans": generated_text.split("<answer>")[-1],
                    "stop_reason_final": "finished",
                    "gen_text_store": gen_text_store + generated_text + "</answer>"
                }
                finished_texts.append(original_data)     
            elif extract_answer_math(generated_text) != '':
                original_data = {
                    "question": question,
                    "answer": answer,
                    "pred_ans": extract_answer_math(generated_text),
                    "stop_reason_final": "finished",
                    "gen_text_store": gen_text_store + generated_text
                }
                finished_texts.append(original_data)
            elif matched:
                begin_token, end_token = matched
                query = generated_text.split(begin_token)[-1].split(end_token)[0]
                query = query.replace('"', "").replace("'", "").replace("\t", " ").replace("...", "")

                if query:
                    query_list.append(query)
                    prev_reasonings.append(gen_text_store + generated_text.split(begin_token)[0])
                    original_data = {
                        "chat_prompt": prompt + generated_text.strip(),
                        "answer": answer,
                        "question": question,
                        "stop_reason": stop_reason,
                        "gen_text_store": gen_text_store + generated_text.strip(),
                    }
                    continued_texts.append(original_data)
                else:
                    original_data = {
                        "question": question,
                        "answer": answer,
                        "gen_text_store": gen_text_store + generated_text.strip(),
                        "generated_text": generated_text,
                        "stop_reason_final": "query_inst_error",
                        "pred_ans": "I don't know."
                    }
                    finished_texts.append(original_data)
            else:
                original_data = {
                    "question": question,
                    "answer": answer,
                    "gen_text_store": gen_text_store + generated_text.strip(),
                    "stop_reason_final": "shot_down",
                    "pred_ans": "I don't know."
                }
                finished_texts.append(original_data)

        print("=="*80)
        if query_list:
            url_wiki = f"http://localhost:{args.port}/queries"
            status_code = 0
            while status_code != 200:
                response = requests.post(url_wiki, json={"queries": query_list, "prev_reasonings": prev_reasonings, "k": args.topk, "sources": ["rag" for _ in range(len(query_list))]})
                status_code = response.status_code
                value = random.uniform(0.01, 0.5)
                time.sleep(value)
            if response.status_code == 200:
                result = response.json()
                answers = result["answers"]
                for i in range(len(answers)):
                    retrieve_docs = answers[i]
                    continued_text_now = copy.deepcopy(continued_texts[i])
                    if len(retrieve_docs) > 0:
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
                    finished_texts.append({
                        "question": continued_texts[i]["question"],
                        "answer": continued_texts[i]["answer"],
                        "gen_text_store": continued_texts[i]["gen_text_store"],
                        "stop_reason_final": "retrieve_error",
                        "pred_ans": "I don't know."
                    })
                continued_texts = []

        finished_all_list.extend(finished_texts)
        print("=="*80)
        print(f"Epoch: {k}, New_Finished: {len(finished_texts)}, All_Finished: {len(finished_all_list)}, Continued: {len(continued_texts)}")
        print("Begin Writing Epoch: ", k)
        print("=="*80)

        if k == 0:
            t = time.localtime()
            if 'hotpot500' in args.src_file:
                output_dir = f'/AIRPFS/lwt/eval/outputs/hotpot500/{model_short_name}.rag'
            elif '2wiki500' in args.src_file:
                output_dir = f'/AIRPFS/lwt/eval/outputs/2wiki500/{model_short_name}.rag'
            elif 'musique500' in args.src_file:
                output_dir = f'/AIRPFS/lwt/eval/outputs/musique500/{model_short_name}.rag'
            elif 'bamboogle' in args.src_file:
                output_dir = f'/AIRPFS/lwt/eval/outputs/bamboogle/{model_short_name}.rag'
            result_json_name = f'{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.jsonl'
            result_path = os.path.join(output_dir, result_json_name)
            os.makedirs(output_dir, exist_ok=True)
        if len(finished_texts) > 0:
            with open(result_path, "a") as f:
                for text in finished_texts:
                    f.write(json.dumps(text) + "\n")
        if len(continued_texts) != 0:
            continued_answer = copy.deepcopy(continued_texts)
        else:
            continued_answer = []
            break

    eval(result_path)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()