import faiss
import pickle
from FlagEmbedding import FlagModel
import os


import argparse
import pickle
import torch
import json

# 假设FlagModel是已经定义的类
# from your_model_file import FlagModel

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
        c_len = len(corpus)
        new_corpus = []
        line_num = 0
        for line in corpus:
            line_num += 1
            if line_num % 10000 == 0:
                print(f"Percent: {line_num}/{c_len}")

            title_text = line.split('\t')[1].strip('')
            new_corpus.append(title_text)
    return new_corpus

def process_corpus(file_path, save_path):
    # 设置使用的GPU
    # if torch.cuda.is_available() and gpu_id >= 0:
    #     torch.cuda.set_device(gpu_id)
    #     print(f"Using GPU: {gpu_id}")
    # else:
    #     print("Using CPU")
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 调用函数加载语料库
    print("Start load corpus")
    corpus = load_corpus(file_path)
    print(f"Load {len(corpus)} from {file_path}.")

    # 打印加载的语料库
    for sample in corpus[:2]:
        print(sample)

    # Load model (automatically use GPUs)
    model = FlagModel('/AIRPFS/lwt/model/bge-large-en-v1.5',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=False)

    print("Start encode")
    corpus_embeddings = model.encode_corpus(corpus, batch_size=3584, max_length=300)

    print("Shape of the corpus embeddings:", corpus_embeddings.shape)
    print("Data type of the embeddings:", corpus_embeddings.dtype)

    print("Start save")
    with open(save_path, 'ab') as f:
        pickle.dump(corpus_embeddings, f)
    print("Save over")

if __name__ == "__main__":
    print("1111")
    parser = argparse.ArgumentParser(description='Process corpus and save embeddings.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input TSV file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output pickle file.')

    args = parser.parse_args()

    process_corpus(args.file_path, args.save_path)
