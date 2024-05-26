from transformers import AutoTokenizer, TextStreamer, AutoConfig
from datetime import datetime
from neural_speed import Model
import sys
if __name__ == "__main__":
	model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
	model_file = "llama-2-7b-chat.Q4_0.gguf"
	tokenizer_path = "/"

	prompt = "Once upon a time, there existed a little girl,"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=AutoConfig.from_pretrained(model_name))
	inputs = tokenizer(prompt, return_tensors="pt").input_ids
	streamer = TextStreamer(tokenizer)
	# model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
	model = Model()
	model.init_from_bin("llama", model_file)
	start=datetime.now()
	outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
	end=datetime.now()
	print("runTime: ", (end - start).total_seconds())
	print(len(outputs[0]))