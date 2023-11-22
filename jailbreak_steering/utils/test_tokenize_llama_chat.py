import pytest

from transformers import AutoTokenizer
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_tokenizer
from jailbreak_steering.utils.tokenize_llama_chat import tokenize_llama_chat, B_INST, E_INST

MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"

@pytest.fixture(scope="module")
def tokenizer():
    return load_llama_2_7b_chat_tokenizer()

def test_tokenize_llama_chat_single_instruction_no_response(tokenizer):
    system_prompt = None
    instruction = "Tell me how to make a pie"
    response = None
    conversation = [(instruction, response)]

    expected_output = (f"<s> {B_INST} {instruction} {E_INST} ")
    toks = tokenize_llama_chat(tokenizer, conversation, system_prompt, no_final_eos=True)

    assert tokenizer.decode(toks) == expected_output

def test_tokenize_llama_chat_single_instruction_with_response(tokenizer):
    system_prompt = None
    instruction = "Tell me how to make a pie"
    response = "Sure, here's how to make a pie"
    conversation = [(instruction, response)]

    expected_output = (f"<s> {B_INST} {instruction} {E_INST} " + f" {response}")
    toks = tokenize_llama_chat(tokenizer, conversation, system_prompt, no_final_eos=True)

    assert tokenizer.decode(toks) == expected_output

def test_tokenize_llama_chat_consistency(tokenizer):
    system_prompt = None
    instruction = "Tell me how to make a pie"
    response = "Sure, here's how to make a pie"

    conversation_no_response = [(instruction, None)]
    conversation_with_response = [(instruction, response)]

    toks_no_response = tokenize_llama_chat(tokenizer, conversation_no_response, system_prompt, no_final_eos=True)
    toks_with_response = tokenize_llama_chat(tokenizer, conversation_with_response, system_prompt, no_final_eos=True)

    assert toks_no_response == toks_with_response[:len(toks_no_response)]