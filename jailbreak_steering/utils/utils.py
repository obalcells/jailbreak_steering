from typing import List
from torch import Tensor
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import torch
import tqdm

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def show_topk(tokenizer: AutoTokenizer, prompts: List[str], logits: Float[Tensor, "batch_size seq_len d_model"], k=10):
    topk_logits, topk_tokens = logits[:, -1, :].topk(k, dim=-1)

    print("Top 10 tokens:")
    for j, prompt in enumerate(prompts):
        topk_string = " "
        for k, tok in enumerate(topk_tokens[j]):
            topk_string += f"({tokenizer.decode(tok, skip_special_tokens=False)}, {tok}, {topk_logits[j][k]}) "
        print(f"\t {prompt} -> {topk_string}")

def generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    toks: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated=64,
    fwd_hooks=[],
    include_prompt=False,
    show_first_topk=False
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    model.reset_hooks()
    
    for i in tqdm.tqdm(range(max_tokens_generated)):
        logits = model.run_with_hooks(
            all_toks[:, :-max_tokens_generated + i],
            return_type="logits",
            fwd_hooks=fwd_hooks,
        )

        # greedy sampling (temperature=0)
        next_tokens = logits[:, -1, :].argmax(dim=-1)

        all_toks[:,-max_tokens_generated+i] = next_tokens

        if show_first_topk and i == 0:
            show_topk(tokenizer, tokenizer.batch_decode(toks), logits, k=10)

    if include_prompt:
        return tokenizer.batch_decode(all_toks)
    else:
        return tokenizer.batch_decode(all_toks[:, toks.shape[1]:])

def instruction_to_prompt(
    instruction: str,
    system_prompt: str=None,
    model_output: str=None,
    append_space: bool=True,
) -> str:
    """
    Converts an instruction to a prompt string structured for Llama2-chat.
    See details of Llama2-chat prompt structure here: here https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    if system_prompt is not None:
        dialog_content = B_SYS + system_prompt.strip() + E_SYS + instruction.strip()
    else:
        dialog_content = instruction.strip()

    if model_output is not None:
        prompt = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    else:
        prompt = f"{B_INST} {dialog_content.strip()} {E_INST}"

    if append_space:
        prompt = prompt + " "

    return prompt

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    padding_length: int=None,
    system_prompt: str=None,
    model_outputs: List[str]=None,
    append_space_after_inst: bool=True
) -> Int[Tensor, "batch seq_len"]:
    if model_outputs is not None:
        assert(len(instructions) == len(model_outputs))
        prompts = [
            instruction_to_prompt(instruction, system_prompt, model_output, append_space=False)
            for (instruction, model_output) in zip(instructions, model_outputs)
        ]
    else:
        prompts = [
            instruction_to_prompt(instruction, system_prompt, model_output=None, append_space=append_space_after_inst)
            for instruction in instructions
        ]

    instructions_toks = tokenizer(
        prompts,
        padding="max_length" if padding_length is not None else True,
        max_length=padding_length,
        truncation=False,
        return_tensors="pt"
    ).input_ids

    return instructions_toks

def generate_from_instructions(
    tl_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_new_tokens: int=64,
    temperature: float=0.0,
):
    instructions_toks = tokenize_instructions(tokenizer, instructions)

    output_ids = tl_model.generate(instructions_toks, max_new_tokens=max_new_tokens, temperature=temperature)
    for answer_idx, answer in enumerate(tokenizer.batch_decode(output_ids)):
        print(f"\nGeneration #{answer_idx}:\n\t{repr(answer)}")