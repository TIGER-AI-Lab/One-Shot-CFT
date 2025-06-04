#!/bin/bash

mkdir ../data
cd ../data

# For Math Reasoning Training, get these four datasets from TIGER-Lab/One-Shot-CFT-Data or directly download by "wget":
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/DSR-CFT-p0.jsonl
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/DSR-CFT-p1.jsonl
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/DSR-CFT-p2.jsonl
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/DSR-CFT-p3.jsonl


# For Logic Reasoning Training, get these  datasets from TIGER-Lab/One-Shot-CFT-Data or directly download by "wget":
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/BBEH-CFT-CausalUnderstanding-p0.jsonl
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/BBEH-CFT-CausalUnderstanding-p1.jsonl

wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/BBEH-CFT-DisambiguationQA-p0.jsonl
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/BBEH-CFT-DisambiguationQA-p1.jsonl

wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/BBEH-CFT-TimeArithmetic-p0.jsonl
wget https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data/resolve/main/BBEH-CFT-TimeArithmetic-p1.jsonl
