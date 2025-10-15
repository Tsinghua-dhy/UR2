from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import os
import requests
import json
from typing import List

# 初始化模型路径映射
MODEL_MAP = {
    "qwen3-4b": "/AIRPFS/lwt/model/qwen3-4b",
    "qwen3-8b": "/AIRPFS/lwt/model/qwen3-8b",
    "qwen3-14b": "/AIRPFS/lwt/model/qwen3-14b",
    "qwen3-32b": "/AIRPFS/lwt/model/qwen3-32b",
    "minicpm4-8b": "/AIRPFS/lwt/model/minicpm4-8b",
}
##replace your model path here


# 默认模型
DEFAULT_MODEL = "qwen3-32b"
MODEL_NAME = MODEL_MAP[DEFAULT_MODEL]

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 检索接口 URL
RETRIEVAL_URL = "http://localhost:5005/queries"

# 定义输入数据模型
class QueryRequest(BaseModel):
    queries: List[str]
    k: int = 3
    prev_reasonings: List[str] = []
    sources: List[str] = []

# 定义 Wikipedia 推理链指令函数
def get_wikipedia_to_reasonchain_instruction(prev_reasoning: str, search_query: str, wikipedia_content: str, prompt_type: str, source: str) -> str:
    if source == "math" and prompt_type == "weak_format":
        return f"""**Task Instruction:**

You are assisting in solving a math problem. You are tasked with reading and analyzing Wikipedia content based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Wikipedia Content**. Your task is to extract accurate and relevant information from the provided Wikipedia content to support or enhance the reasoning process.

- Carefully read the provided **Wikipedia Content**;
- Extract factual information that can:
    1. Directly assist in answering the **Current Search Query**, or
    2. Help validate, complete, or correct earlier reasoning steps.

- The extracted information should be:
    1. Accurate and trustworthy;
    2. Closely relevant to the query;
    3. Helpful in improving, expanding, or supporting the mathematical reasoning.

Important:  
Do NOT attempt to correct or rewrite the previous reasoning. Treat it only as contextual reference that may be flawed.

**Output Format:**
Present the information beginning with the label `**Final Information**` as shown below.

**Final Information**

[Helpful factual information]

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Wikipedia Content:**  
{wikipedia_content}

Now you should analyze the Wikipedia content and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""
    elif source == "math" and prompt_type == "strong_format":
        return f"""**Task Instruction:**

You are assisting in solving a math problem. Your task is to determine whether the current query requires external factual knowledge (such as definitions, formulas, theorems, or lookup values), and if so, extract accurate and relevant information from the provided Wikipedia content to support or enhance the reasoning process.

**Step 1: Classify the Query Type**

Determine whether the query falls into one of the following categories:

- **Knowledge-based query**: Can be directly answered using factual knowledge (e.g., definitions, known formulas, theorems, constants, or table lookups).
- **Reasoning-based query**: Requires multi-step deduction, logical reasoning, or constructive computation, and cannot be directly answered by retrieval alone.

- If the query is knowledge-based, proceed to Step 2.
- If the query is reasoning-based, return the following message under **Final Information**:
```
  This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning.
```

**Step 2: Analyze Knowledge-Based Queries (if applicable)**

If the query is classified as knowledge-based:

- Carefully read the provided **Wikipedia Content**;
- Extract factual information that can:
    1. Directly assist in answering the **Current Search Query**, or
    2. Help validate, complete, or correct earlier reasoning steps.

- The extracted information should be:
    1. Accurate and trustworthy;
    2. Closely relevant to the query;
    3. Helpful in improving, expanding, or supporting the mathematical reasoning.

Important:  
Do NOT attempt to correct or rewrite the previous reasoning. Treat it only as contextual reference that may be flawed.

**Output Format:**
Present the information beginning with the label `**Final Information**` as shown below.

**Final Information**

[Helpful factual information, or the non-knowledge-based response]

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Wikipedia Content:**  
{wikipedia_content}

Now determine whether the query "{search_query}" is knowledge-based or reasoning-based.  
If knowledge-based, extract and present helpful information from Wikipedia.  
If not, return the specified response.
"""
    elif prompt_type == "weak_format":
        return f"""**Task Instruction:**

You are tasked with reading and analyzing Wikipedia content based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Wikipedia Content**. Your objective is to extract factual and relevant information from the **Wikipedia Content** that directly supports or informs the **Current Search Query**, and integrate it into the reasoning process in an objective and helpful manner.

**Guidelines:**

1. **Analyze the Wikipedia Content:**
   - Carefully read the Wikipedia content.
   - Identify factual information that directly relates to the **Current Search Query** and may assist in refining or expanding the reasoning.

2. **Maintain Objectivity:**
   - Do not attempt to validate, confirm, or correct the **Previous Reasoning Steps**.
   - Treat them as context only—useful for understanding, but potentially flawed.
   - Your role is to supplement the reasoning neutrally with accurate, relevant facts from Wikipedia.

3. **Output Format:**
   - Present the helpful information for the current search query beginning with `**Final Information**` as shown below.

**Final Information**

[Helpful information]

---
**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Wikipedia Content:**  
{wikipedia_content}

Now you should analyze the Wikipedia content and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""
    elif prompt_type == "strong_format":
        return f"""**Task Instruction:**

Your first task is to determine whether the provided query is a knowledge-based query that can be answered using factual information from Wikipedia, or if it is a query requiring design, computation, or complex reasoning (e.g., creating a plan, performing calculations, or generating creative content). 

**Step 1: Query Classification**
- If the query is knowledge-based (e.g., asking for facts, definitions, or historical information), proceed to analyze the Wikipedia content and extract relevant information.
- If the query is not knowledge-based (e.g., it involves designing something, performing computations, or requires subjective reasoning), immediately return the following response under **Final Information**:
  ```
  This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning.
  ```

**Step 2: Analyze Knowledge-Based Queries (if applicable)**
If the query is knowledge-based:
- Carefully read the Wikipedia content.
- Identify factual information that directly relates to the **Current Search Query** and may assist in refining or expanding the reasoning.
- Maintain objectivity: Do not attempt to validate, confirm, or correct the **Previous Reasoning Steps**. Treat them as context only—useful for understanding, but potentially flawed.
- Your role is to supplement the reasoning neutrally with accurate, relevant facts from Wikipedia.

**Output Format:**
- Present the helpful information for the current search query beginning with `**Final Information**` as shown below.

**Final Information**

[Helpful information or the non-knowledge-based response]

---
**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Wikipedia Content:**  
{wikipedia_content}

Now you should first classify the query "{search_query}" as knowledge-based or non-knowledge-based. If knowledge-based, analyze the Wikipedia content and find helpful information. If non-knowledge-based, return the specified response.
"""
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'weak_format' or 'strong_format'.")

# 批量 Qwen3 总结请求函数（一次性调用）
def qwen3_batch_summarize(queries: List[str], wikipedia_contents_list: List[List[str]], model_type: str, prev_reasonings: List[str], max_tokens: int, enable_thinking: bool, prompt_type: str, sources: List[str]) -> List[str]:
    model = MODEL_MAP.get(model_type, MODEL_NAME)
    prompts = []
    
    # 为每个查询生成 prompt
    for query, contents, prev_reasoning, source in zip(queries, wikipedia_contents_list, prev_reasonings, sources):
        wiki_text = '\n\n'.join([f'Wikipedia Content {i+1}: {content}' for i, content in enumerate(contents)])
        prompt = get_wikipedia_to_reasonchain_instruction(prev_reasoning, query, wiki_text, prompt_type, source)
        messages = [{"role": "user", "content": prompt}]
        if model_type == "minicpm4-8b":
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        prompts.append(prompt_text)

    # 配置生成参数
    if model_type == "minicpm4-8b":
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.95,
            top_k=20,
            min_p=0,
            repetition_penalty=1.02
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=max_tokens,
            top_p=0.7,
            top_k=10,
            min_p=0,
            presence_penalty=1.0
        )

    # 批量生成
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"Error in Qwen3 batch summarization: {str(e)}")
        return ["Error summarizing Wikipedia content for this query."] * len(queries)

    # 处理生成结果
    results = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        if enable_thinking:
            response = response.split("</think>")[-1].strip()
        final_info_start = "**Final Information**"
        if final_info_start in response:
            final_info = response.split("**Final Information**")[-1].split("## Final Information")[-1].split("##Final Information")[-1].strip().replace("```","").strip()[:3000]
            if "This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning." in final_info:
                final_info = "This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning."
            results.append(final_info)
        else:
            results.append("No Final Information found in Qwen3 response.")

    return results

# 创建 FastAPI 应用
app = FastAPI()

@app.post("/queries")
async def query(request: QueryRequest):
    # 从请求中获取查询数据
    try:
        queries = request.queries
        k = request.k
        prev_reasonings = request.prev_reasonings or [""] * len(queries)
        sources = request.sources or ["default_source"] * len(queries)
        
        # 验证输入
        if not queries:
            raise HTTPException(status_code=400, detail="No queries provided")
        if len(prev_reasonings) != len(queries):
            raise HTTPException(status_code=400, detail="Mismatch between queries and previous reasoning steps")
        if len(sources) != len(queries):
            # 如果 sources 长度不匹配，用默认值填充或截取
            if len(sources) == 1:
                sources = sources * len(queries)
            else:
                sources = (sources + ["default_source"] * len(queries))[:len(queries)]

        # 记录查询开始时间
        search_start = time.time()

        # 构造符合第一个服务期望的请求格式，保持与原始queries相同的顺序
        retrieval_queries = []
        for i, query in enumerate(queries):
            retrieval_queries.append({
                "query": query,
                "source": sources[i],
                "k": k
            })

        # 调用检索接口获取 Wikipedia 内容
        try:
            retrieval_payload = {"queries": retrieval_queries}
            #print(f"Sending retrieval request: {retrieval_payload}")
            
            retrieval_response = requests.post(
                RETRIEVAL_URL,
                json=retrieval_payload,
                timeout=30
            )
            retrieval_response.raise_for_status()
            retrieval_data = retrieval_response.json()
            
            #print(f"Received retrieval response: {retrieval_data}")
            
            # 从返回的结果中提取 answers，确保顺序与输入queries一致
            if "results" in retrieval_data:
                # 检查返回结果的顺序是否与输入一致
                if len(retrieval_data["results"]) != len(queries):
                    raise HTTPException(status_code=500, detail=f"Mismatch: expected {len(queries)} results, got {len(retrieval_data['results'])}")
                
                # 验证查询顺序一致性（可选的调试步骤）
                for i, result in enumerate(retrieval_data["results"]):
                    if "query" in result and result["query"] != queries[i]:
                        print(f"Warning: Query order mismatch at index {i}. Expected: '{queries[i]}', Got: '{result.get('query', 'N/A')}'")
                
                wikipedia_contents = []
                for result in retrieval_data["results"]:
                    if "answers" in result:
                        wikipedia_contents.append(result["answers"])
                    else:
                        wikipedia_contents.append([])
            else:
                raise HTTPException(status_code=500, detail="Invalid response format from retrieval service")
                
        except requests.RequestException as e:
            print(f"Retrieval request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve Wikipedia content: {str(e)}")

        # 验证检索结果
        if len(wikipedia_contents) != len(queries):
            raise HTTPException(status_code=500, detail=f"Mismatch between queries ({len(queries)}) and retrieved Wikipedia content ({len(wikipedia_contents)})")

        # 批量调用 Qwen3 总结 Wikipedia 内容，顺序已经由检索服务保证
        summaries = qwen3_batch_summarize(
            queries=queries,
            wikipedia_contents_list=wikipedia_contents,
            model_type=model_type,
            prev_reasonings=prev_reasonings,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            prompt_type=prompt_type,
            sources=sources
        )

        # 创建输出目录
        output_dir = "./qwen3_summarize_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存查询、Wikipedia 内容和总结到 JSONL 文件
        output_file = os.path.join(output_dir, f"summarize_results.jsonl")
        with open(output_file, 'a', encoding='utf-8') as f:
            for query, contents, summary, prev_reasoning in zip(queries, wikipedia_contents, summaries, prev_reasonings):
                json.dump(
                    {"query": query, "wikipedia_contents": contents, "prev_reasoning": prev_reasoning, "answers": summary},
                    f,
                    ensure_ascii=False
                )
                f.write('\n')

        # 计算查询耗时
        search_end = time.time()
        search_time = search_end - search_start
        avg_search_time = search_time / len(queries) if queries else 0
        print(f"Total processing time: {search_time:.4f}s, Average per query: {avg_search_time:.4f}s")

        # 返回 JSON 格式的响应，确保与输入查询顺序一致
        return {"queries": queries, "answers": [[summary] for summary in summaries]}

    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process queries: {str(e)}")

if __name__ == "__main__":
    import sys
    import uvicorn
    # 默认参数
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5004
    model_type = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    gpu_ids = sys.argv[3].split(",") if len(sys.argv) > 3 else ["0"]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    prompt_type = sys.argv[5] if len(sys.argv) > 5 else "strong_format"
    enable_thinking = sys.argv[6].lower() == "true" if len(sys.argv) > 6 else True

    # 验证模型类型
    if model_type not in MODEL_MAP:
        print(f"Invalid model_type: {model_type}. Supported: {list(MODEL_MAP.keys())}")
        sys.exit(1)

    # 设置 GPU 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using GPUs: {gpu_ids}")

    # 初始化模型
    model_path = MODEL_MAP.get(model_type, MODEL_NAME)
    if model_type == "minicpm4-8b":        
        llm = LLM(model=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=len(gpu_ids), trust_remote_code=True, max_num_batched_tokens=32768)
    else:
        llm = LLM(model=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=len(gpu_ids))
    print(f"Summarization interface started, port: {port}, model: {model_type}, GPU IDs: {gpu_ids}, "
          f"Thinking: {enable_thinking}, Max Tokens: {max_tokens}")

    # 启动 FastAPI 应用
    uvicorn.run(app, host="localhost", port=port)