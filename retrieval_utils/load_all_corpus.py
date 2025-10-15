import argparse
import os
import time
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from FlagEmbedding import FlagModel
import uvicorn
import json

# 定义单个查询的模型
class SingleQuery(BaseModel):
    query: str
    source: str = "default_source"  # 每个查询的来源，允许为 null
    k: int = 3  # 每个查询的 k 值

# 定义请求模型
class QueryRequest(BaseModel):
    queries: List[SingleQuery]

# 初始化 FastAPI 应用
app = FastAPI()

# 加载语料库函数
def load_corpus(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        if "abstract" in file_path:
            corpus = [line.strip() for line in file.readlines()]
        elif "wiki" in file_path:
            corpus = [line.split('\t')[1].strip() for line in file.readlines()]
        elif "med" in file_path:
            corpus = [json.loads(line)["content"] for line in file.readlines()]
    return corpus

# 全局变量：存储多个语料库的模型、索引和语料
corpus_dict: Dict[str, List[str]] = {}
index_dict: Dict[str, faiss.Index] = {}
model = None

def get_soure(source):
    if source == "rag":
        return "wiki_abstract"
    else:
        return "wiki_full"

# 初始化语料库和索引
def initialize_corpus_and_index(corpus_configs: List[Dict[str, str]], default_source: str):
    global model, corpus_dict, index_dict

    # 验证 default_source
    valid_sources = [config["source"] for config in corpus_configs]
    if default_source not in valid_sources:
        raise ValueError(f"Invalid default_source: {default_source}. Must be one of {valid_sources}")

    # 初始化模型
    model = FlagModel(
        '/AIRPFS/lwt/model/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False
    )
    print("Model loaded successfully")

    # 加载所有语料库和索引
    for config in corpus_configs:
        source = config["source"]
        corpus_path = config["corpus_path"]
        index_path = config["index_path"]

        # 加载语料库
        corpus_dict[source] = load_corpus(corpus_path)
        print(f"Corpus loaded for {source} - {len(corpus_dict[source])} entries")

        # 加载 FAISS 索引
        index_dict[source] = faiss.read_index(index_path)
        print(f"Index loaded for {source}")
        print(f"Index type: {type(index_dict[source])}")
        print(f"Is trained: {index_dict[source].is_trained}")
        print(f"Total vectors: {index_dict[source].ntotal}")
        print(f"Dimension: {index_dict[source].d}")

    print("All corpora and indices loaded successfully")
    print(f"Default source: {default_source}")
    print("Ready to accept queries")

@app.post("/queries")
async def query(request: QueryRequest):
    global model, corpus_dict, index_dict
    try:
        # 为每个查询创建一个带索引的处理项，保持原始顺序
        query_items = []
        for i, q in enumerate(request.queries):
            source = get_soure(q.source) if q.source != "default_source" else args.default_source
            if source not in index_dict:
                raise HTTPException(status_code=400, detail=f"Invalid source: {source}. Available sources: {list(index_dict.keys())}")
            query_items.append({
                'original_index': i,
                'query': q.query,
                'source': source,
                'k': q.k
            })
            if q.source is None:
                print(f"Query '{q.query}' has no source specified, using default source: {source}")

        # 按source分组，但保持每个查询的原始索引
        source_to_queries = {}
        for item in query_items:
            source = item['source']
            if source not in source_to_queries:
                source_to_queries[source] = []
            source_to_queries[source].append(item)

        # 初始化结果数组，确保与输入查询顺序一致
        all_results = [None] * len(request.queries)

        # 处理每个source的查询
        for source, items in source_to_queries.items():
            index = index_dict[source]
            corpus = corpus_dict[source]

            # 配置索引参数（仅对 HNSW 索引）
            if hasattr(index, 'hnsw'):
                index.hnsw.efConstruction = 1024
                index.hnsw.efSearch = 2048

            # 提取查询和相关信息
            queries = [item['query'] for item in items]
            k_values = [item['k'] for item in items]
            original_indices = [item['original_index'] for item in items]
            max_k = max(k_values)

            # 编码查询
            query_embeddings = model.encode_queries(queries)
            print(f"Query embeddings shape for {source}: {query_embeddings.shape}")

            # 搜索
            search_start = time.time()
            D, I = index.search(query_embeddings, k=max_k)
            search_end = time.time()

            search_time = search_end - search_start
            avg_search_time = search_time / len(queries)
            print(f"Index search total time for {source}: {search_time:.4f}s, average per query: {avg_search_time:.4f}s")

            # 将结果放回正确的位置
            for i, (query, k, orig_idx) in enumerate(zip(queries, k_values, original_indices)):
                answers = [corpus[idx] for idx in I[i][:k]]
                all_results[orig_idx] = {
                    "query": query,
                    "answers": answers,
                    "source": source,
                    "k": k
                }

        return {"results": all_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run FastAPI server with multiple FAISS indices and corpora")
    parser.add_argument("port", type=int, help="Port to run the FastAPI server")
    parser.add_argument("default_source", type=str, choices=["wiki_full", "wiki_abstract", "medicine"], help="Default source for queries with no source specified")
    parser.add_argument("--wiki_full_index", type=str, default="/AIRPFS/lwt/corpus/enwiki_kilt_all_hnsw.bin", help="Path to Wikipedia full index file")
    parser.add_argument("--wiki_full_corpus", type=str, default="/AIRPFS/lwt/corpus/wiki_kilt_100_really.tsv", help="Path to Wikipedia full corpus file")
    parser.add_argument("--wiki_abstract_index", type=str, default="/AIRPFS/lwt/corpus/enwiki_2017_abstract_hnsw.bin", help="Path to Wikipedia abstract index file")
    parser.add_argument("--wiki_abstract_corpus", type=str, default="/AIRPFS/lwt/corpus/enwiki_2017_abstract.tsv", help="Path to Wikipedia abstract corpus file")
    parser.add_argument("--medicine_index", type=str, default="/AIRPFS/lwt/corpus/med_corpus.bin", help="Path to Medicine index file")
    parser.add_argument("--medicine_corpus", type=str, default="/AIRPFS/lwt/corpus/med_corpus.jsonl", help="Path to Medicine corpus file")
    args = parser.parse_args()

    # 配置语料库
    corpus_configs = [
        {"source": "wiki_full", "corpus_path": args.wiki_full_corpus, "index_path": args.wiki_full_index},
        {"source": "wiki_abstract", "corpus_path": args.wiki_abstract_corpus, "index_path": args.wiki_abstract_index},
        #{"source": "medicine", "corpus_path": args.medicine_corpus, "index_path": args.medicine_index}
    ]

    # 初始化语料库和索引
    initialize_corpus_and_index(corpus_configs, args.default_source)

    # 运行 FastAPI 服务器
    uvicorn.run(app, host="localhost", port=args.port)