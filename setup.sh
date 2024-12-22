#!/bin/bash

echo "Downloading 3-Bit Quantized Llama 3.2 from huggingface.co"

mkdir data

sudo wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ3_M.gguf

python3 -m venv venv

source venv/bin/activate

python3 -m pip install requests langchain_community pypdf sentence_transformers pandas

git clone git@github.com:ggerganov/llama.cpp.git

cd llama.cpp

mkdir build

cd build

cmake ..

make

cd ../../



