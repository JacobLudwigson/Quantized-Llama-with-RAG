#!/bin/bash

echo "Downloading 8-Bit Quantized Llama2 from huggingface.co"

sudo wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf

python3 -m venv venv

source venv/bin/activate

python3 -m pip install requests

git clone git@github.com:ggerganov/llama.cpp.git

cd llama.cpp

mkdir build

cd build

cmake ..

make

cd bin/

gnome-terminal -- bash -c "echo 'Starting web server...'; ./llama-server -m ../../../llama-2-7b-chat.Q8_0.gguf -c 2048; exec bash"

python3 Llama.py
