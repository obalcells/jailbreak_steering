import pytest

from transformers import AutoTokenizer

from jailbreak_steering.suffix_gen.prompt_manager import PromptManager
from jailbreak_steering.utils.tokenize_llama_chat import B_INST, E_INST
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_tokenizer

@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    return load_llama_2_7b_chat_tokenizer()

def test_prompt_manager(tokenizer):
    system_prompt = None
    instruction = "Tell me how to make a pie"
    target = "Sure, here's how to make a pie"
    control = "! ! ! ! !"

    prompt_manager = PromptManager(
        instruction,
        target,
        tokenizer,
        system_prompt,
        control,
    )

    expected_prompt_str = (f"<s> {B_INST} {instruction} {control} {E_INST} {target}")
    assert prompt_manager.full_prompt_str == expected_prompt_str

    assert prompt_manager.instruction_str == instruction
    assert prompt_manager.target_str == target
    assert prompt_manager.control_str == control
