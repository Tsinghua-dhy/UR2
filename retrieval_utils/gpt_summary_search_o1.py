from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx
import json
from datetime import datetime
import os
import requests
from tqdm import tqdm
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 支持的模型映射
MODEL_MAP = {
    "4omini": "gpt-4o-mini-2024-07-18",
    "3.5turbo": "gpt-3.5-turbo-0125",
    "4.1": "gpt-4.1-2025-04-14",
    "4.1mini": "gpt-4.1-mini-2025-04-14",
    "4.1nano": "gpt-4.1-nano-2025-04-14",
}

# 检索接口 URL
RETRIEVAL_URL = "http://localhost:5005/queries"

# 定义 OpenAI 客户端配置

#Replace your OpenAI API key here
API_CONFIGS = {
    "77": {
        "api_key": os.getenv("OPENAI_API_KEY_77", ""),
        "base_url": "https://api.key77qiqi.com/v1",
    },
    "xty": {
        "api_key": os.getenv("OPENAI_API_KEY_XTY", ""),
        "base_url": "https://svip.xty.app/v1",
    }
}

# 初始化 OpenAI 客户端的函数
def initialize_openai_client(key_source: str) -> OpenAI:
    if key_source not in API_CONFIGS:
        raise ValueError(f"Invalid key_source: {key_source}. Must be one of {list(API_CONFIGS.keys())}")
    
    config = API_CONFIGS[key_source]
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        http_client=httpx.Client(
            base_url=config["base_url"],
            follow_redirects=True,
        ),
    )

# 定义请求数据模型，添加 sources 字段以支持分流
class QueryRequest(BaseModel):
    queries: List[str]
    k: int = 3
    prev_reasonings: List[str] = None
    sources: List[str] = None

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

# 单个 GPT 总结请求函数
def gpt_summarize(query: str, wikipedia_contents: List[str], model_type: str, prev_reasoning: str, prompt_type: str, source: str, client: OpenAI) -> str:
    model = MODEL_MAP.get(model_type, "gpt-4o-mini-2024-07-18")
    wiki_text = '\n\n'.join([f'Wikipedia Content {i+1}: {content}' for i, content in enumerate(wikipedia_contents)])
    prompt = get_wikipedia_to_reasonchain_instruction(prev_reasoning, query, wiki_text, prompt_type, source)
    t = 0
    while t < 5:  # 重试次数
        t += 1
        try:
            rst = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                timeout=20
            )
            response = rst.choices[0].message.content.strip()
            final_info_start = "**Final Information**"
            if final_info_start in response:
                final_info = response[response.index(final_info_start) + len(final_info_start):].replace("```","").strip()
                if "This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning." in final_info:
                    final_info = "This query requires design, computation, or complex reasoning, which exceeds the capabilities of a search engine. Please input another query or proceed with direct reasoning."
                return final_info
            else:
                return "No Final Information found in GPT response."
        except Exception as e:
            logger.error(f"Error in GPT summarization for query '{query}': {str(e)}")
            time.sleep(1)
            continue
    return "Failed to give Wikipedia content, possibly due to political sensitivity, retrieval blocking, or summarization errors caused by restricted content."
# 批量 GPT 总结请求函数
def gpt_batch_summarize(queries: List[str], wikipedia_contents_list: List[List[str]], model_type: str, prev_reasonings: List[str], prompt_type: str, sources: List[str], client: OpenAI, max_threads: int = 128) -> List[str]:
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_idx = {
            executor.submit(gpt_summarize, query, contents, model_type, prev_reasoning, prompt_type, source, client): idx
            for idx, (query, contents, prev_reasoning, source) in enumerate(zip(queries, wikipedia_contents_list, prev_reasonings, sources))
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Summarizing queries"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = "Error summarizing Wikipedia content for this query."
    return results

# 创建 FastAPI 应用
app = FastAPI()

@app.post("/queries")
async def query(data: QueryRequest):
    client = app.state.client  # 直接从 app.state 获取客户端
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    prev_reasonings = data.prev_reasonings if data.prev_reasonings is not None else [""] * len(data.queries)
    sources = data.sources if data.sources is not None else ["default_source"] * len(data.queries)

    if not data.queries:
        raise HTTPException(status_code=400, detail="No queries provided")
    if len(prev_reasonings) != len(data.queries):
        raise HTTPException(
            status_code=400,
            detail="Mismatch between queries and previous reasoning steps"
        )
    if len(sources) != len(data.queries):
        sources = (sources + ["default_source"] * len(data.queries))[:len(data.queries)]

    search_start = time.time()
    retrieval_queries = [{"query": query, "source": source, "k": data.k} for query, source in zip(data.queries, sources)]

    try:
        retrieval_response = requests.post(
            RETRIEVAL_URL,
            json={"queries": retrieval_queries},
            timeout=30
        )
        retrieval_response.raise_for_status()
        retrieval_data = retrieval_response.json()

        if "results" in retrieval_data:
            if len(retrieval_data["results"]) != len(data.queries):
                raise HTTPException(
                    status_code=500,
                    detail=f"Mismatch: expected {len(data.queries)} results, got {len(retrieval_data['results'])}"
                )
            wikipedia_contents = [result["answers"] if "answers" in result else [] for result in retrieval_data["results"]]
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from retrieval service")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Wikipedia content: {str(e)}")

    summaries = gpt_batch_summarize(data.queries, wikipedia_contents, app.state.model_type, prev_reasonings, app.state.prompt_type, sources, client)

    output_dir = "./gpt_summarize_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"summarize_results.jsonl")
    with open(output_file, 'a', encoding='utf-8') as f:
        for query, contents, summary, prev_reasoning, source in zip(data.queries, wikipedia_contents, summaries, prev_reasonings, sources):
            json.dump(
                {
                    "query": query,
                    "wikipedia_contents": contents,
                    "prev_reasoning": prev_reasoning,
                    "source": source,
                    "answers": summary
                },
                f,
                ensure_ascii=False
            )
            f.write('\n')

    search_end = time.time()
    search_time = search_end - search_start
    avg_search_time = search_time / len(data.queries) if data.queries else 0
    logger.info(f"Total processing time: {search_time:.4f}s, Average per query: {avg_search_time:.4f}s")

    outs = [[summary] for summary in summaries]
    return {"queries": data.queries, "answers": outs}

if __name__ == "__main__":
    import uvicorn

    # 获取命令行参数
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5004
    model_type = sys.argv[2] if len(sys.argv) > 2 else "4omini"
    prompt_type = sys.argv[3] if len(sys.argv) > 3 else "strong_format"
    key_source = sys.argv[4] if len(sys.argv) > 4 else "77"

    # 验证 prompt_type 和 key_source
    if prompt_type not in ["weak_format", "strong_format"]:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'weak_format' or 'strong_format'.")
    if key_source not in API_CONFIGS:
        raise ValueError(f"Invalid key_source: {key_source}. Must be one of {list(API_CONFIGS.keys())}")

    # 将 model_type 和 prompt_type 存储在 app.state 中
    app.state.model_type = model_type
    app.state.prompt_type = prompt_type

    # 初始化 OpenAI 客户端
    try:
        client = initialize_openai_client(key_source)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # 将客户端存储在 app.state 中以便在请求中使用
    app.state.client = client

    logger.info(f"Summarization interface started, port: {port}, model: {model_type}, prompt_type: {prompt_type}, key_source: {key_source}")
    uvicorn.run(app, host="localhost", port=port)