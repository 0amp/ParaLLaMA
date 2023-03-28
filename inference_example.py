# %%
import llama
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# %%
MODEL = 'decapoda-research/llama-7b-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

# %%
tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
# %%
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True)
model.half()
model.cuda()
# %%
batch = tokenizer("2 times 2 is 4. 4 times 4 is 16. 16 times 16 is", return_tensors = "pt")
print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=45)[0]))

# Expected output: "Yo mama is so fat, she has to buy two seats on the plane"
# %%
model.eval()
with torch.no_grad():
    output, head_hidden_states, mlp_hidden_states = model(batch["input_ids"].cuda(), output_head_hidden_states = True, output_hidden_states = True)
# %%
