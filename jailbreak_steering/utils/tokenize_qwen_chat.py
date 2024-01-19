from typing import List
import torch

from transformers import AutoTokenizer

# Qwen chat templates are based on
# - Official examples from Qwen repo: https://github.com/QwenLM/Qwen/blob/5aa84bdfd3237b37f01bc88cd49b3279b9a71d0b/examples/vllm_wrapper.py#L32
# - Online guidelines: https://github.com/Ki-Seki/chat_prompt_templates?tab=readme-ov-file#qwen-prompt-template

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def format_qwen_chat(
    instruction: str,
    target: str=None,
    system: str=None,
    include_trailing_newline: bool=True
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
    if not include_trailing_newline:
        formatted_instruction = formatted_instruction.strip()
    
    return f"{formatted_instruction}{target}"

def tokenize_qwen_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    target: str=None,
    system: str=None,
    include_trailing_newline=True
):
    prompt = format_qwen_chat(
        instruction=instruction,
        target=target,
        system=system,
        include_trailing_newline=include_trailing_newline
    )

    toks = tokenizer.encode(prompt)
    
    return toks