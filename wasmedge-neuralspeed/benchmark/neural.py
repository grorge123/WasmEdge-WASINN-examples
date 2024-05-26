from transformers import AutoTokenizer, TextStreamer
#from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from datetime import datetime
from neural_speed import Model
# Specify the GGUF repo on the Hugginface
model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
# Download the the specific gguf model file from the above repo
model_file = "llama-2-7b-chat.Q4_0.gguf"
# make sure you are granted to access this model on the Huggingface.
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"

prompt = "Once upon a time, there existed a little girl,"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
# model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
model = Model()
model.init_from_bin("llama", "/home/george/LFX-mentorship/tokenizersRust/llama-2-7b-chat.Q4_0.gguf")
start=datetime.now()
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
end=datetime.now()
print("runTime: ", (end - start).total_seconds())
print(len(outputs[0]))
