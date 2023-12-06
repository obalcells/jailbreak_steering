from typing import List, Tuple, Optional

# Default system prompt from https://huggingface.co/blog/llama2#how-to-prompt-llama-2
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Based off of https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py
def tokenize_llama_chat(
    tokenizer,
    conversation: List[Tuple[str, Optional[str]]],
    system_prompt: str=DEFAULT_SYSTEM_PROMPT,
    no_final_eos: bool=True,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    system_prompt: the system prompt to use for the conversation
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """

    if system_prompt == "default":
        system_prompt = DEFAULT_SYSTEM_PROMPT

    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ) -> List[int]:

        if is_first_message and system_prompt and len(system_prompt) > 0:
            system_content = B_SYS + system_prompt + E_SYS
        else:
            system_content = ""

        instruction_content = system_content + instruction.strip()

        if model_output and len(model_output) > 0:
            if no_eos:
                response_content = model_output.strip()
            else:
                response_content = f"{model_output.strip()} {tokenizer.eos_token}"
            
            return tokenizer.encode(f"{B_INST} {instruction_content} {E_INST} {response_content}")
        else:
            return tokenizer.encode(f"{B_INST} {instruction_content} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens

# performs a binary search to look for a target string inside some tokens, returns a slice of where the target's tokens are located
def find_string_in_tokens(target, tokens, tokenizer) -> slice:
    assert target in tokenizer.decode(tokens), "The target isn't contained in the whole array of tokens"
    # we first perform the binary search over the end index of the slice
    end_idx_left, end_idx_right = 0, len(tokens) 
    while end_idx_left != end_idx_right:
        mid = (end_idx_right + end_idx_left) // 2
        if target in tokenizer.decode(tokens[:mid]):
            end_idx_right = mid
        else:
            end_idx_left = mid + 1
    end_idx = end_idx_left
    # now we perform the binary search over the start index of the slice
    start_idx_left, start_idx_right = 0, end_idx-1 
    while start_idx_left != start_idx_right:
        mid = (start_idx_right + start_idx_left + 1) // 2
        if target in tokenizer.decode(tokens[mid:end_idx]):
            start_idx_left = mid
        else:
            start_idx_right = mid-1
    start_idx = start_idx_left
    target_slice = slice(start_idx, end_idx)
    assert target in tokenizer.decode(tokens[target_slice])
    return target_slice

def get_ascii_toks(tokenizer):
    return [
        token_id for token_id in range(3, tokenizer.vocab_size)
        if tokenizer.decode([token_id]).isascii() and tokenizer.decode([token_id]).isprintable()
    ]

def format_instruction_answer_llama_chat(
    tokenizer,
    conversation: List[Tuple[str, Optional[str]]],
    system_prompt: str=DEFAULT_SYSTEM_PROMPT,
    no_final_eos: bool=True,
) -> str:
    """
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    system_prompt: the system prompt to use for the conversation

    Returns: a string that can be tokenized and passed to the model 
    """

    if system_prompt == "default":
        system_prompt = DEFAULT_SYSTEM_PROMPT

    def _format_instruction_response(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ) -> str:
        if is_first_message and system_prompt and len(system_prompt) > 0:
            system_content = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        else:
            system_content = ""

        instruction_content = system_content + instruction.strip()

        if model_output and len(model_output) > 0:
            if no_eos:
                response_content = model_output.strip()
            else:
                response_content = f"{model_output.strip()} {tokenizer.eos_token}"

            return f"{B_INST} {instruction_content} {E_INST} {response_content}"
        else:
            return f"{B_INST} {instruction_content} {E_INST}"

    prompt = ""
    for i, (user_input, model_output) in enumerate(conversation):
        prompt += _format_instruction_response(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return prompt 