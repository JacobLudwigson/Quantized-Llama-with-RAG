#!/bin/bash
if [ -d "data" ]; then
    echo "data directory already present, skipping mkdir"
else
    mkdir data
fi
if [ -f "Llama-3.2-3B-Instruct-IQ3_M.gguf" ]; then
    echo "Llama already downloaded, skipping wget"
else
    echo "Downloading 3-Bit Quantized Llama 3.2 from huggingface.co"
    sudo wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ3_M.gguf
fi
if [ -d "venv" ]; then
	echo "skipping environment creation, venv already present"
else
	python3 -m venv venv;
	source venv/bin/activate;
	python3 -m pip install requests langchain_community pypdf sentence_transformers pandas;
fi
if [ -d "llama.cpp" ]; then
	echo "Skipping cloning llama.cpp, already present!"
else
	git clone git@github.com:ggerganov/llama.cpp.git;
fi
if [ -d "llama.cpp/build" ]; then
	echo "Skipping llama.cpp build, already built!"
else
	cd llama.cpp;
	mkdir build;
	cd build;
	cmake ..;
	make;
fi
