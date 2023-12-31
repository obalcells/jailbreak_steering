import os
import torch

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA_2_7B_CHAT_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"

def load_llama_2_7b_chat_model():
    load_dotenv()
    huggingface_token = os.environ["HF_TOKEN"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7B_CHAT_MODEL_PATH,
        token=huggingface_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
    ).to(device)

    return model

def load_llama_2_7b_chat_tokenizer():
    load_dotenv()
    huggingface_token = os.environ["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_2_7B_CHAT_MODEL_PATH,
        token=huggingface_token,
        use_fast=True
    )
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    return tokenizer