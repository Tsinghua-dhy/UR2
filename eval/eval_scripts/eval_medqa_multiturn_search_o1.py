import json
import torch
import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import time
import copy
import random
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on MedQA dataset with multi-round retrieval.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_json', type=str, required=True, help="Path to MedQA JSONL file.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of questions to evaluate (-1 for all).")
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=1536)
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="5007")
    parser.add_argument("--max_rounds", type=int, default=7, help="Maximum number of retrieval rounds.")
    return parser.parse_args()

def get_cot_prompt(question, options, model_short_name, tokenizer, args):
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    option_str = "\n".join([f"{l}. {o}" for l, o in zip(option_letters[:len(options)], options)])
    if 'it' in model_short_name or 'instruct' in model_short_name:
        prompt_instruct_retrieve = """You are solving a multiple-choice question. Think step by step and reason carefully before answering.

    During your reasoning, if you’re unsure about any fact, you may issue a **search query** like this:
    `<|begin_of_query|> your concise query <|end_of_query|>`

    * You can issue **multiple queries** at different steps in your reasoning.
    * **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

    Once documents are returned in this format:
    `<|begin_of_documents|> ... <|end_of_documents|>`
    use them to refine your reasoning before proceeding.

    Use this output format:

    ```
    <think> Step-by-step reasoning, including option analysis and retrieved information. </think>
    <answer> Final answer. Use only the letter (A, B, C, etc.). </answer>
    ```
    """
        prompt_instruct_retrieve_0_2_2 = """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

During your reasoning, if you’re unsure about any fact, you may issue a **search query** like this:
<|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

* You can issue **multiple queries** at different steps in your reasoning.
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

Once documents are returned in this format:
<|begin_of_documents|> ... <|end_of_documents|>

use them to refine your reasoning before proceeding and continue reasoning immediately after reviewing the retrieved information.

At the end of your reasoning, give your final answer in the following format:
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""
        prompt_instruct_v0_1_3 = """You are an expert in solving multiple-choice questions and must analyze each option carefully and logically.
        You must reason step by step, explicitly considering the meaning and implications of *each* option, eliminating incorrect choices with clear justification, and identifying the best answer through careful comparison.
        The reasoning process and final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively:
        "<think> [Detailed reasoning here, including analysis of each option: 'Option A... Option B... etc.'] </think>\n\n<answer> [Final answer: A, B, C, ...] </answer>"""
        Question = f"Question: {question}\nOptions:\n{option_str}"
        prompt_instruct_v1_0_0 = """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

The reasoning process should include detailed considerations such as analyzing the question, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps when needed.

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<search> your concise query (less than 20 words) </search>

* You can issue **multiple queries** at different steps in your reasoning.  
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

Once documents are returned in this format:  
<information>  
... (search results here)  
</information>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the options until you're confident in your answer.

At the end of your reasoning, give your final answer in the following format:  
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""

## prompt of ur2 here
        prompt_instruct_v0_3_17 = """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

* You can issue **multiple queries** at different steps in your reasoning.  
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.
  * ✅ Example:
    * <|begin_of_query|> What are the common symptoms of pneumonia? <|end_of_query|>
    * <|begin_of_query|> What is the typical treatment for pneumonia in elderly patients? <|end_of_query|>
  * ❌ Do **not** combine them like this:
    * <|begin_of_query|> What are the symptoms and treatments for pneumonia in elderly patients? <|end_of_query|>
* You may issue **at most four queries** in total — use them wisely.

Once documents are returned in this format:  
<|begin_of_documents|>   
... (search results here)  
<|end_of_documents|>

use the retrieved documents to verify, reject, or revise your prior reasoning about the options. Then continue analyzing the options until you're confident in your answer.

At the end of your reasoning, give your final answer in the following format:  
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""
        prompt_instruct_vsetag = """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<search> your concise query (less than 20 words) </search>

* You can issue **multiple queries** at different steps in your reasoning.  
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.
  * ✅ Example:
    * <search> What are the common symptoms of pneumonia? </search>
    * <search> What is the typical treatment for pneumonia in elderly patients? </search>
  * ❌ Do **not** combine them like this:
    * <search> What are the symptoms and treatments for pneumonia in elderly patients? </search>
* You may issue **at most four queries** in total — use them wisely.

Once documents are returned in this format:  
<information>   
... (search results here)  
</information>

use the retrieved documents to verify, reject, or revise your prior reasoning about the options. Then continue analyzing the options until you're confident in your answer.

At the end of your reasoning, give your final answer in the following format:  
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""
        prompt_instruct_search_o1 = f"""
You are a medical reasoning assistant with the ability to perform web searches to help 
you answer complex clinical and biomedical questions accurately. You have special tools:

- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.
Then, the system will search and analyze relevant web pages or medical resources, and provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {args.max_rounds}.

Once you have all the information you need, continue your reasoning to answer the medical question.

Example:
Question: "Which class of drug is most appropriate for managing diabetic nephropathy in hypertensive patients?"
Assistant thinking steps:
- I should check current guidelines or studies about drugs used for diabetic nephropathy in hypertensive patients.

Assistant:
<|begin_search_query|>first-line treatment for diabetic nephropathy with hypertension<|end_search_query|>

(System returns processed information from relevant medical sources)

Assistant continues reasoning with the new information...

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|.
- When done searching, continue your clinical reasoning based on the evidence.
"""

        if model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.1.3":
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_v0_1_3},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True) + "<think>"
        elif model_short_name == "qwen-2.5-7b-instruct":
            # search-o1 way
            Question = f"Question: {question}\nOptions:\n{option_str}"
            user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format: '
            'the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option).'
            f'Question:\n{Question}\n\n')
            prompt = tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt_instruct_search_o1 + user_prompt},
                    ], tokenize=False, add_generation_prompt=True)
            return prompt
            # ori
            # prompt = tokenizer.apply_chat_template([
            #             {"role": "system", "content": prompt_instruct_retrieve},
            #             {"role": "user", "content": Question},
            #         ], tokenize=False, add_generation_prompt=True) + "<think>"
        elif model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.17" or model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.20" or model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.19" or model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.21" or model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.23":
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_v0_3_17},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)
        elif "qwen-2.5-7b-instruct-rl-ours-vablation-setag" in model_short_name:
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_vsetag},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)
        elif "qwen-2.5-7b-instruct-rl-ours-vablation" in model_short_name:
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_v0_3_17},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)              
        elif "qwen-2.5-3b-instruct-rl-ours-mmlu-v0.3.1" in model_short_name:
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_v0_3_17},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)    
        elif "qwen-2.5-7b-instruct-rl-ours-v1.1.1" in model_short_name:
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_v0_3_17},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)                 
        elif "qwen-2.5-7b-instruct-rl-ours-v1" in model_short_name:
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_v1_0_0},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)
        else:
            prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": prompt_instruct_retrieve_0_2_2},
                        {"role": "user", "content": Question},
                    ], tokenize=False, add_generation_prompt=True)            
    else:
        base_prompt_r1_search_0_1_4 = """The User asks a question, and the Assistant solves it.
            The Assistant is required to solve it step by step, thinking carefully before answering.
            The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> If the question is multiple-choice, only output the corresponding uppercase letter (e.g., A, B, C, D). If it is not multiple-choice, output the complete answer normally. </answer>".
            During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
            Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".
            The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.  The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

            User: {question}
            Options:\n{option_str}
            Assistant: <think>"""
        base_prompt_r1_search = """The User asks a question, and the Assistant solves it.
    The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
    The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
    During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
    Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nOptions:\n{option_str}\nAssistant: <think>"""
        base_prompt_search_r1 ="""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: {question}\nOptions:\n{option_str}\n"""
        prompt_v_0_3 = """The User asks a question, and the Assistant solves it.  
    The Assistant is required to solve it step by step, thinking carefully before answering.  
    The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,  
    "<think> reasoning process here </think>\n\n<answer> If the question is multiple-choice, only output the corresponding uppercase letter (e.g., A, B, C, D). If it is not multiple-choice, output the complete answer normally. </answer>".

    During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary, using the format:  
    "<|begin_of_query|> keyword_1 keyword_2 ... <|end_of_query|>".  

    **Each query must correspond to only ONE triple or factual statement.**  
    Avoid comma-separated lists or conjunctions like "and"/"or" that include multiple ideas.  
    If multiple pieces of information are needed, issue multiple separate queries.

    Then, the search system will provide the Assistant with retrieval information in the format:  
    "<|begin_of_documents|> ...search results... <|end_of_documents|>".  

    The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.  
    The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

    User: {question}  
    Options:\n{option_str}  
    Assistant: <think>"""
        prompt_v_0_1_0 = """The User asks a question, and the Assistant solves it.
    The Assistant is required to solve it step by step, thinking carefully before answering.
    The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> If the question is multiple-choice, only output the corresponding uppercase letter (e.g., A, B, C, D). If it is not multiple-choice, output the complete answer normally. </answer>".
    User: {question}
    Options:\n{option_str}
    Assistant: <think>"""
        prompt_v_0_1_1 = """The User asks a question, and the Assistant solves it. The Assistant is an expert in solving multiple-choice questions and must analyze each option carefully and logically.

    The Assistant must reason step by step, explicitly considering the meaning and implications of *each* option, eliminating incorrect choices with clear justification, and identifying the best answer through careful comparison.

    The reasoning process and final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively:
    "<think> [Detailed reasoning here, including analysis of each option: 'Option A... Option B... etc.'] </think>\n\n<answer> [Final answer: A, B, C, ...] </answer>"

    User: {question}
    Options:\n{option_str}
    Assistant: <think>"""
        if model_short_name == "qwen-2.5-7b-base-rl-ours-v0.2.0":
            prompt = base_prompt_r1_search_0_1_4.format(question=question,option_str=option_str)
        elif model_short_name == "qwen-2.5-7b-r1-search" or "qwen-2.5-7b-base-rl-ours-v0" in model_short_name:
            prompt = base_prompt_r1_search.format(question=question,option_str=option_str)
        elif model_short_name == "qwen-2.5-7b-search-r1" or model_short_name == "qwen-2.5-7b-search-r1-it":
            prompt = base_prompt_search_r1.format(question=question,option_str=option_str)
        elif model_short_name == "qwen-2.5-7b-base-rl-ours-mmlu-v0.1" or model_short_name == "qwen-2.5-7b-base-rl-ours-mmlu-v0.2" or model_short_name == "qwen-2.5-7b-base-rl-ours-mmlu-v0.0":
            prompt = base_prompt_r1_search_0_1_4.format(question=question,option_str=option_str)
        elif model_short_name == "qwen-2.5-7b-base-rl-ours-mmlu-v0.1.0":
            prompt = prompt_v_0_1_0.format(question=question,option_str=option_str)
        elif model_short_name == "qwen-2.5-7b-base-rl-ours-mmlu-v0.1.1":
            prompt = prompt_v_0_1_1.format(question=question,option_str=option_str)
        else:
            prompt = prompt_v_0_3.format(question=question,option_str=option_str)
    return prompt

SEARCH_FORMATS = {
    "qwen-2.5-7b-r1-search": {
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
    "qwen-2.5-7b-instruct-rl-ours-vablation-setag2": {
        "begin_token": "<search>",
        "end_token": "</search>",
        "doc_begin": "<information>",
        "doc_end": "</information>",
    },
    "qwen-2.5-7b-instruct-rl-ours-v1.0.0":{
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
    ## search token of ur2 here
}

def match_search_format(generated_text, stop_reason, model_short_name):
    if model_short_name not in SEARCH_FORMATS:
        model_short_name = "qwen-2.5-7b-r1-search"
    format_info = SEARCH_FORMATS[model_short_name]
    begin, end = format_info["begin_token"], format_info["end_token"]
    if begin in generated_text and stop_reason == end:
        return begin, end
    return None

def get_doc_insertion_text(model_short_name, doc_content):
    if model_short_name not in SEARCH_FORMATS:
        model_short_name = "qwen-2.5-7b-r1-search"
    format_info = SEARCH_FORMATS[model_short_name]
    return (
        f"{format_info['end_token']}\n\n"
        f"{format_info['doc_begin']}\n{doc_content}{format_info['doc_end']}\n\n"
    )

def normalize_text(text):
    """Normalize text by converting to lowercase and replacing spaces with underscores."""
    return re.sub(r'\s+', '_', text.lower().strip())

import re

def extract_answer_math(s):
    extracted_text = ''
    patterns = [
        r'\\boxed\{(.*)\}',  # 匹配 LaTeX 格式的 \boxed{...}
        r'<answer>(.*?)</answer>',  # 匹配 <answer>...</answer>
        r'[Tt]he correct answer is:?\s*([A-Ja-j])',  # 匹配 the/The correct answer is[:] A-J/a-j
        r'Final answer\s*:?\s*([A-Ja-j])'  # 新增：匹配 Final answer[:] A-J/a-j
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, s, re.DOTALL | re.IGNORECASE))
    
    if matches:
        extracted_text = matches[-1]  # 取最后一个匹配
    return extracted_text.strip()

def extract_choice(text, options):
    """
    Extract the predicted answer by checking for a single letter or matching option content.
    
    Args:
        text (str): The model's output text.
        options (list): List of option strings (e.g., ["Stress fracture of proximal tibia", ...]).
    
    Returns:
        str: A single letter (A, B, C, D, etc.) corresponding to the predicted answer, or None if not found.
    """
    text = text.strip()
    
    # 1. Check for explicit answer formats (e.g., "Final Answer: D", "correct answer is: D")
    patterns = [
        r'\\boxed{([A-J])}',  # Matches LaTeX-style "\boxed{D}"
        r'Final Answer[:\s]*([A-J])',  # Matches "Final Answer: D" or "Final Answer D"
        r'correct answer is[:\s]*([A-J])',  # Matches "correct answer is: D" or "correct answer is D"
        r'\[Answer\][\s:]*([A-J])',  # Matches "[Answer] D" or "[Answer]: D"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 2. Check for option letter followed by text (e.g., "D. Nystatin")
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, ...
        # Look for patterns like "D. Option text" or "D: Option text"
        option_pattern = rf'{letter}[\.\:]\s*{re.escape(option)}'
        if re.search(option_pattern, text, re.IGNORECASE):
            return letter
    
    # 3. Fallback: Normalize and match option content
    normalized_options = [normalize_text(opt) for opt in options]
    normalized_text = normalize_text(text)
    
    for i, norm_opt in enumerate(normalized_options):
        if norm_opt in normalized_text:
            return chr(65 + i)  # Convert index to letter (0 -> A, 1 -> B, etc.)
    
    # 4. Fallback: Check for single letter at the end of the text
    last_chars = text[-5:].strip() if len(text) > 5 else text.strip()
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        if letter in last_chars and len(last_chars) <= 2:  # Ensure it's a single letter
            return letter
    
    # 5. Return None if no answer is found
    return None

def evaluate_medqa(llm, tokenizer, data, args):
    model_short_name = args.model_path.split("/")[-1].lower()
    topk = args.topk
    if args.subset_num != -1:
        data = data[:args.subset_num]

    continued_answer = []

    for item in data:
        question = item['question']
        options = item['options']
        correct_idx = item["answer_idx"] if item.get("answer_idx") is not None else item["answer"]
        prompt = get_cot_prompt(question, options, model_short_name, tokenizer, args)
        continued_answer.append({
            "chat_prompt": prompt,
            "question": question,
            "options": options,
            "answer": correct_idx,
            "gen_text_store": ""
        })

    if model_short_name == "qwen-2.5-7b-instruct":
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_search_query|>", "</answer>"]
    elif model_short_name == "qwen-2.5-7b-r1-search" or "qwen-2.5-7b-base-rl-ours-v0" in model_short_name or model_short_name=="qwen-2.5-7b-instruct-rl-ours-v1.1.1":
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
    elif model_short_name == "qwen-2.5-7b-instruct-rl-ours-vablation-setag2":
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>"]        
    elif "qwen-2.5-7b-search-r1" in model_short_name:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>", "</answer>"]
    elif "qwen-2.5-7b-instruct-rl-ours-v1" in model_short_name:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>"]
    elif model_short_name == "qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.17":
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>"]
    else:
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop=stop_tokens
    )

    finished_all_list = []
    for k in range(args.max_rounds):
        prompts = [item["chat_prompt"] for item in continued_answer]
        if not prompts:
            break
        outputs = llm.generate(prompts, sampling_params)
        finished_texts = []
        continued_texts = []
        query_list = []
        prev_reasonings = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            stop_reason = output.outputs[0].stop_reason
            current_data = continued_answer[i]
            gen_text_store = current_data["gen_text_store"]

            matched = match_search_format(generated_text, stop_reason, model_short_name)
            if extract_answer_math(generated_text) != '':
                pred_ans = extract_choice(generated_text, current_data["options"])
                finished_texts.append({
                    "question": current_data["question"],
                    "options": current_data["options"],
                    "answer": current_data["answer"],
                    "gen_text_store": gen_text_store + generated_text,
                    "stop_reason_final": "finished",
                    "pred_ans": pred_ans
                })
            elif matched:
                begin_token, end_token = matched
                query = generated_text.split(begin_token)[-1].split(end_token)[0]
                query = query.replace('"', "").replace("'", "").replace("\t", " ").replace("...", "")
                if query:
                    query_list.append(query)
                    prev_reasonings.append(gen_text_store + generated_text.split(begin_token)[0])
                    continued_texts.append({
                        "chat_prompt": current_data["chat_prompt"] + generated_text.strip(),
                        "question": current_data["question"],
                        "options": current_data["options"],
                        "answer": current_data["answer"],
                        "gen_text_store": gen_text_store + generated_text.strip()
                    })
                else:
                    finished_texts.append({
                        "question": current_data["question"],
                        "options": current_data["options"],
                        "answer": current_data["answer"],
                        "gen_text_store": gen_text_store + generated_text.strip(),
                        "stop_reason_final": "query_inst_error",
                        "pred_ans": "I don't know."
                    })
            elif k == args.max_rounds - 1:
                finished_texts.append({
                    "question": current_data["question"],
                    "options": current_data["options"],
                    "answer": current_data["answer"],
                    "gen_text_store": gen_text_store + generated_text.strip(),
                    "stop_reason_final": "max_rounds",
                    "pred_ans": "I don't know."
                })
                continue
            else:
                finished_texts.append({
                    "question": current_data["question"],
                    "options": current_data["options"],
                    "answer": current_data["answer"],
                    "gen_text_store": gen_text_store + generated_text.strip(),
                    "stop_reason_final": "shot_down",
                    "pred_ans": "I don't know."
                })

        # 模拟检索（实际使用时需替换为真实检索服务）
        if query_list:
            url_wiki = f"http://localhost:{args.port}/queries"
            try:
                status_code = 0
                while status_code != 200:
                    try:
                        response = requests.post(url_wiki, json={"queries": query_list, "prev_reasonings": prev_reasonings, "k": topk, "sources": ["medicine" for _ in query_list]})
                        status_code = response.status_code
                        if status_code != 200:
                            time.sleep(random.uniform(0.01, 0.5))  # Retry after a short delay
                    except Exception as e:
                        print(f"Request error: {e}")
                        time.sleep(random.uniform(0.1, 1.0))  # Retry after a longer delay

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
                            "options": continued_texts[i]["options"],
                            "answer": continued_texts[i]["answer"],
                            "gen_text_store": continued_texts[i]["gen_text_store"],
                            "stop_reason_final": "retrieve_error",
                            "pred_ans": "I don't know."
                        })
                    continued_texts = []
            except Exception as e:
                print(f"Retrieval error: {e}")
                for i in range(len(continued_texts)):
                    finished_texts.append({
                        "question": continued_texts[i]["question"],
                        "options": continued_texts[i]["options"],
                        "answer": continued_texts[i]["answer"],
                        "gen_text_store": continued_texts[i]["gen_text_store"],
                        "stop_reason_final": "retrieve_error",
                        "pred_ans": "I don't know."
                    })
                continued_texts = []
        finished_all_list.extend(finished_texts)
        print(
            f"Epoch: {k}, New Finished: {len(finished_texts)}, All Finished: {len(finished_all_list)}, Continued: {len(continued_texts)}")
        continued_answer = copy.deepcopy(continued_texts)

    # 处理最终结果
    results = []
    correct_count = 0
    for item in finished_all_list:
        pred = item["pred_ans"] if item.get("pred_ans") else None
        gt = item["answer"]
        is_correct = pred == gt if pred is not None else False
        if is_correct:
            correct_count += 1
        results.append({
            "question": item["question"],
            "options": item["options"],
            "gt_answer": gt,
            "pred_answer": pred,
            "is_correct": is_correct,
            "full_output": item["gen_text_store"],
            "stop_reason": item["stop_reason_final"]
        })

    acc = correct_count / len(data) if results else 0.0
    print(f"[✓] Accuracy: {acc:.2%}")
    return acc, results

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_rate,
    )
    model_short_name = args.model_path.split("/")[-1]
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    acc, results = evaluate_medqa(llm, tokenizer, data, args)
    if "mmlupro_med" in args.input_json:
        output_dir = f'../outputs/mmlupro_med/{model_short_name}.rag'
    else:
        output_dir = f'../outputs/medqa/{model_short_name}.rag'
    os.makedirs(output_dir, exist_ok=True)
    t = time.localtime()
    with open(os.path.join(output_dir, f"{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json"), 'w') as f:
        json.dump({"accuracy": f"{acc * 100:.2f}"}, f, indent=2)
    with open(os.path.join(output_dir, f"{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()