from openai import OpenAI
import httpx
import json
import concurrent.futures
import functools
import requests
from typing import Literal
from transformers import AutoTokenizer
import time
##目前4的key不可用
import asyncio
import aiohttp
import tiktoken
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed



def truncate_prompt(prompt, max_tokens):
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 适用于 gpt-3.5-turbo 和 gpt-4
  truncated_tokens = encoding.encode(prompt)[:max_tokens]
  return encoding.decode(truncated_tokens)

api_key = "sk-vWYNncOlkjNuQZfDjkCaZWYe1yUkUs9bMxHuWem9RHsNzR3j"
client = OpenAI(
    base_url="https://svip.xty.app/v1",
    api_key=api_key,
    http_client=httpx.Client(
        base_url="https://svip.xty.app/v1",
        follow_redirects=True,
    ),
)
api_4_key = "sk-vWYNncOlkjNuQZfDjkCaZWYe1yUkUs9bMxHuWem9RHsNzR3j"
client_4 = OpenAI(
    base_url="https://svip.xty.app/v1",
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url="https://svip.xty.app/v1",
        follow_redirects=True,
    ),
)
def llama3_request(prompt, temp=0.0): 
  url = f"http://localhost:{8091}/v1"
  prompt = [{
        "role": "user",
        "content": prompt
    }]
  client = OpenAI(
        base_url=url,
        api_key="token-abc123",
    )
  tokenizer = AutoTokenizer.from_pretrained("/disks/disk0/private/kaiming/ckpts/Llama-3.1-8B-Instruct")
  prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
  while(True):
    try:
      completion = client.completions.create(
                model="/disks/disk0/private/kaiming/ckpts/Llama-3.1-8B-Instruct",
                prompt=prompt,
                temperature=temp,
                max_tokens=4096,
            )
      return completion.choices[0].text.strip()
    except Exception as e:
      print(f"Error when calling deepseek: {e}")
      time.sleep(5)

def qwen_request(prompt, temp = 0.0): 
  url = f"http://localhost:8092/v1"
  prompt = [{
        "role": "user",
        "content": prompt
    }]
  client = OpenAI(
        base_url=url,
        api_key="token-abc123",
    )
  tokenizer = AutoTokenizer.from_pretrained("/disks/disk0/private/kaiming/ckpts/Qwen2.5-7B-Instruct")
  prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
  while(True):
    try:
      completion = client.completions.create(
                model="qwen",
                prompt=prompt,
                temperature=temp,
                max_tokens=4096,
            )
      return completion.choices[0].text.strip()
    except Exception as e:
      print(f"Error when calling deepseek: {e}")
      time.sleep(5)

def ChatGPT_request(prompt,temperature=0): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  while(True):
    try:
      encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 适用于 gpt-3.5-turbo 和 gpt-4
      if len(encoding.encode(prompt)) > 15700:
        prompt = truncate_prompt(prompt, 15700)
      rst = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=temperature,
        max_tokens = 512,
        messages=[
          {"role": "user", "content": prompt}
        ]
      )
      return rst.choices[0].message.content.strip()
    except Exception as e:
      print(f"Error when calling deepseek: {e}")
      time.sleep(5)
def GPT_Instruct_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  while(True):
    try:
      rst = client.completions.create(
          model="gpt-3.5-turbo-instruct",
          temperature=0.0,
          max_tokens = 4096,
          prompt=prompt
      )
      return rst.choices[0].text.strip()
    except:
      print("ChatGPT ERROR")

# 单个请求函数
def gpt4omini_request(prompt):
    while True:
        try:
            rst = client_4.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                temperature=0.0,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return rst.choices[0].message.content.strip()
        except Exception as e:
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

def GPT4o_request(prompt): 
  '''
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  '''
  # temp_sleep()
  while(True):
    try:
      rst = client_4.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        max_tokens = 512,
        messages=[
          {"role": "user", "content": prompt}
        ]
      )
      return rst.choices[0].message.content.strip()
    except:
      print("ChatGPT ERROR")
  '''
  while(True):
    try:
      url = "http://15.204.101.64:4000/v1/chat/completions"
      headers = {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer sk-UOArhyzuKw4Xaiga3e40F22502B44a6c93CaAaC336A3A1F1'
      }
      data = {
          "model": "gpt-4o",
          "messages": [{"role": "user",  "content": prompt}],
          "temperature": 0.0,
          "max_tokens": 4096,
          "stream": False}
      response = requests.post(url, json=data, headers=headers)
      response = response.json()
      return response["choices"][0]["message"]["content"].strip()
    except:
      print("ChatGPT ERROR")
  '''

def GPT4_request(prompt): 
  '''
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  while(True):
    try:
      rst = client_4.chat.completions.create(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens = 4096,
        messages=[
          {"role": "user", "content": prompt}
        ]
      )
      return rst.choices[0].message.content.strip()
    except:
      print("ChatGPT ERROR")'''
  while(True):
    try:
      url = "http://15.204.101.64:4000/v1/chat/completions"
      headers = {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer sk-UOArhyzuKw4Xaiga3e40F22502B44a6c93CaAaC336A3A1F1'
      }
      data = {
          "model": "gpt-4-0613",
          "messages": [{"role": "user",  "content": prompt}],
          "temperature": 0.0,
          "max_tokens": 4096,
          "stream": False}
      response = requests.post(url, json=data, headers=headers)
      response = response.json()
      return response["choices"][0]["message"]["content"].strip()
    except:
      print("ChatGPT ERROR")

def GPT4_Turbo_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  '''while(True):
    try:
      rst = client_4.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.0,
        max_tokens = 4096,
        messages=[
          {"role": "user", "content": prompt}
        ]
      )
      return rst.choices[0].message.content.strip()
    except:
      print("ChatGPT ERROR")
  return rst.choices[0].message.content.strip()
  '''
  while(True):
    try:
      url = "http://15.204.101.64:4000/v1/chat/completions"
      headers = {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer sk-UOArhyzuKw4Xaiga3e40F22502B44a6c93CaAaC336A3A1F1'
      }
      data = {
          "model": "gpt-4-turbo",
          "messages": [{"role": "user",  "content": prompt}],
          "temperature": 0.0,
          "max_tokens": 4096,
          "stream": False}
      response = requests.post(url, json=data, headers=headers)
      response = response.json()
      return response["choices"][0]["message"]["content"].strip()
    except:
      print("ChatGPT ERROR")

def run(topk, res_file, case_file, process_slice):
    topk = int(topk)    
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 10
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        # 并行评测八份切片
        results = []
        process_slice = functools.partial(process_slice, topk = topk)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_slice, slices)
        # 合并八份切片的结果
        for result in results:
            final_result.extend(result)
        with open(res_file, "w", encoding = "utf-8" ) as json_file:
            json.dump(final_result, json_file,  ensure_ascii=False, indent=4)
#print(GPT4o_request("hello, give me a introduction of yourself, are you gpt-4o?"))
