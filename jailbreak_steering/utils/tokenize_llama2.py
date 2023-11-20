from typing import List, Tuple, Optional

# Default system prompt from https://huggingface.co/blog/llama2#how-to-prompt-llama-2
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Borrowed from https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py
def tokenize_llama_chat(
    tokenizer,
    conversation: List[Tuple[str, Optional[str]]],
    system_prompt: str=DEFAULT_SYSTEM_PROMPT,
    no_final_eos=False,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    system_prompt: the system prompt to use for the conversation
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """

    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ):
        if system_prompt and len(system_prompt) > 0:
            system_content = B_SYS + system_prompt + E_SYS
        else:
            system_content = ""

        if is_first_message:
            dialog_content = system_content + instruction.strip()
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )
            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        else:
            return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens