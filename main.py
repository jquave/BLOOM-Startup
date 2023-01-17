from transformers import AutoTokenizer, BloomModel, BloomForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
model = BloomForCausalLM.from_pretrained("bigscience/bloom").cuda()


prompt = """
Introduce yourself.
"""
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=len(prompt)+128,
    )
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)