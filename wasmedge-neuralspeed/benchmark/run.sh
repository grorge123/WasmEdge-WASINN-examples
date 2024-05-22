#!/bin/bash

TEST_TIMES=${TEST_TIMES:-1}

for i in $(seq 1 $TEST_TIMES); do
    timestamp=$(date +%s)
    /main -m /llama-2-7b-chat.Q4_0.gguf -p "Once upon a time, there existed a little girl," --enable-log > /output/ggml-cpp-$timestamp.log 2>&1
	/.wasmedge-neuralspeed/bin/wasmedge --dir .:. --nn-preload default:NeuralSpeed:AUTO:llama-2-7b-chat.Q4_0.gguf /wasmedge-neural-speed.wasm default > /output/ggml-wasi-$timestamp.log 2>&1
	/.wasmedge-ggml/bin/wasmedge --dir .:. --env enable-log=1 --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q4_0.gguf /wasmedge-ggml-llama.wasm default "Once upon a time, there existed a little girl," > /output/neural-wasi-$timestamp.log 2>&1
	python3 /neural.py > /output/neural-py-$timestamp.log 2>&1
	

done

mv *.log /output