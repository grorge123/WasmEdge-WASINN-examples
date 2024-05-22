FROM wasmedge/wasmedge:latest

RUN apt update && apt upgrade
RUN apt -y install python3-dev python3-pip git
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add wasm32-wasi
RUN wget https://raw.githubusercontent.com/intel/neural-speed/main/requirements.txt
RUN pip install -r requirements.txt
RUN pip install neural-speed==1.0
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp
RUN cd /llama.cpp && cmake -B build && cmake --build build --config Release
RUN mv /llama.cpp/build/bin/main /
RUN git clone -b neural_speed https://github.com/grorge123/WasmEdge.git /WasmEdge 
RUN cd /WasmEdge \
	&& cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_BUILD_TESTS=OFF -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="neuralspeed" -DCMAKE_INSTALL_PREFIX=/.wasmedge-neuralspeed \
	&& cmake --build build \
	&& cmake --install build
RUN cd /WasmEdge \ 
	&& rm -rf build \
	&& cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_BUILD_TESTS=OFF -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="ggml" -DCMAKE_INSTALL_PREFIX=/.wasmedge-ggml \
	&& cmake --build build \
	&& cmake --install build
RUN git clone -b benchmark https://github.com/grorge123/WasmEdge-WASINN-examples.git /WasmEdge-WASINN-examples
RUN git clone -b neuralspeed https://github.com/grorge123/wasmedge-wasi-nn.git /WasmEdge-WASINN-examples/wasmedge-wasi-nn
RUN cd /WasmEdge-WASINN-examples/wasmedge-neuralspeed/ && cargo build --target wasm32-wasi --release && mv /WasmEdge-WASINN-examples/wasmedge-neuralspeed/target/wasm32-wasi/release/wasmedge-neural-speed.wasm /
RUN mv /WasmEdge-WASINN-examples/wasmedge-neuralspeed/llama-tokenizer.json /tokenizer.json
RUN mv /WasmEdge-WASINN-examples/wasmedge-neuralspeed/benchmark/neural.py /
RUN mv /WasmEdge-WASINN-examples/wasmedge-ggml/llama/wasmedge-ggml-llama.wasm /
RUN wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf -P /
ENV NEURAL_SPEED_VERBOSE=1
ENV TEST_TIMES=1

RUN mv /WasmEdge-WASINN-examples/wasmedge-neuralspeed/benchmark/run.sh /

RUN mkdir /output

CMD ["bash", "/run.sh"]