#!/bin/bash

# 设置默认值
PORT=${1:-5005}
DEFAULT_SOURCE=${2:-wiki_full}

# 设置 CUDA 环境变量
export CUDA_VISIBLE_DEVICES=0,7

# 运行 Python 脚本
# replace your corpus and hnsw file here
python load_all_corpus.py $PORT $DEFAULT_SOURCE \
    --wiki_full_index /AIRPFS/lwt/corpus/enwiki_kilt_all_hnsw.bin \
    --wiki_full_corpus /AIRPFS/lwt/corpus/wiki_kilt_100_really.tsv \
    --wiki_abstract_index /AIRPFS/lwt/corpus/enwiki_2017_abstract_hnsw.bin \
    --wiki_abstract_corpus /AIRPFS/lwt/corpus/enwiki_2017_abstract.tsv \
    #--medicine_index /AIRPFS/lwt/corpus/med_corpus.bin \
    #--medicine_corpus /AIRPFS/lwt/corpus/med_corpus.jsonl