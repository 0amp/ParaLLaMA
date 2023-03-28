import torch
import transformers
import llama
from llama.tokenization_llama import LLaMATokenizer
from llama.modeling_llama import LLaMADecoderLayer
from llama.parallama import LLaMAPolicy
from datasets import load_dataset
from utils import tokenized_tqa, get_llama_activations
from tqdm import tqdm
from parallelformers.policies.base import Policy, Layer
from parallelformers.utils import AllReduceLinear
from parallelformers import parallelize

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(): 
    """
    Demonstrates running larger LLaMA models split across multiple GPUs
    with large contexts using a port to Parallelformers. 
    """

    # MODEL = 'decapoda-research/llama-7b-hf'
    MODEL = 'decapoda-research/llama-13b-hf'
    # MODEL = 'decapoda-research/llama-30b-hf'
    # MODEL = 'decapoda-research/llama-65b-hf'

    tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
    model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True)
    r = model.half()
    r = model.eval()
    parallelize(model, num_gpus=4, fp16=True, custom_policies=[LLaMAPolicy])

    print(model.dtype)

    batch = tokenizer("2 to the 10th power is 1024.\n2 to the 11th power is", return_tensors = "pt")
    print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=75)[0]))


if __name__ == "__main__": 
    main()