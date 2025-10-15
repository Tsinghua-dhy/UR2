import argparse
import os
import time
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from FlagEmbedding import FlagModel

# Define request model
class QueryRequest(BaseModel):
    queries: List[str]
    k: int = 3

# Initialize FastAPI app
app = FastAPI()

# Load corpus function
def load_corpus(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = [line.strip("\n") for line in file.readlines()]
    return corpus

# Global variables
model = None
corpus = None
index = None

@app.post("/queries")
async def query(request: QueryRequest):
    global index, corpus, model
    try:
        # Configure index parameters
        index.hnsw.efConstruction = 1024
        index.hnsw.efSearch = 2048

        # Encode queries
        query_embeddings = model.encode_queries(request.queries)
        print(f"Query embeddings shape: {query_embeddings.shape}")

        all_answers = []
        search_start = time.time()
        D, I = index.search(query_embeddings, k=request.k)
        search_end = time.time()

        search_time = search_end - search_start
        avg_search_time = search_time / len(request.queries)
        print(f"Index search total time: {search_time:.4f}s, average per query: {avg_search_time:.4f}s")

        # Collect answers
        for idx in I:
            answers_for_query = [corpus[i] for i in idx[:request.k]]
            all_answers.append(answers_for_query)

        return {"queries": request.queries, "answers": all_answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to run the FastAPI server")
    parser.add_argument("index_path", type=str, help="Path to the FAISS index file")
    parser.add_argument("corpus_path", type=str, help="Path to the corpus file")
    args = parser.parse_args()

    # Initialize model
    model = FlagModel(
        '/AIRPFS/lwt/model/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False
    )
    print("Model loaded successfully")

    # Load corpus
    corpus = load_corpus(args.corpus_path)
    print(f"Corpus loaded - {len(corpus)} entries")

    # Load FAISS index
    index = faiss.read_index(args.index_path)
    print("Index loaded successfully")
    print(f"Index type: {type(index)}")
    print(f"Is trained: {index.is_trained}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")
    print("Ready to accept queries")

    # Run FastAPI server
    import uvicorn
    uvicorn.run(app, host="localhost", port=args.port)