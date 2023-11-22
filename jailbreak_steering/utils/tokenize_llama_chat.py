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

    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ) -> List[int]:

        # NOTE:
        # For the instructions, we add a space after the E_INST token.
        # When we tokenize the instruction on its own, the space is encoded as token 29871.
        # When we tokenize instruction + response, the space is not encoded as a token.
        # Thus, to make the tokenization consistent, we tokenize the instruction and response separately,
        # and then concatenate them.        

        if is_first_message and system_prompt and len(system_prompt) > 0:
            system_content = B_SYS + system_prompt + E_SYS
        else:
            system_content = ""

        instruction_content = system_content + instruction.strip()

        instruction_toks = tokenizer.encode(f"{B_INST} {instruction_content.strip()} {E_INST} ")

        if model_output and len(model_output) > 0:
            if no_eos:
                response_content = model_output.strip()
            else:
                response_content = f"{model_output.strip()} {tokenizer.eos_token}"

            response_toks = tokenizer.encode(
                response_content,
                add_special_tokens=False # don't append <s>
            )
        else:
            response_toks = []

        return instruction_toks + response_toks

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens