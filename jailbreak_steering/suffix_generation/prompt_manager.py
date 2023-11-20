import torch

from transformers import AutoTokenizer

from jailbreak_steering.suffix_generation.suffix_generation_utils import find_string_in_tokens
from jailbreak_steering.utils.tokenize_llama2 import tokenize_llama_chat, E_INST

class PromptManager(object):
    """A class used to manage the prompt during optimization."""
    def __init__(
        self,
        instruction:str,
        target: str,
        tokenizer: AutoTokenizer,
        system_prompt: str,
        control_init: str,
    ):
        self.instruction = instruction
        self.target = target
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

        self.control = control_init

        self._update_ids()

    def _update_ids(self):
        toks = tokenize_llama_chat(
            self.tokenizer,
            conversation=[(f"{self.instruction} {self.control}", self.target)],
            system_prompt=self.system_prompt,
            no_final_eos=True,
        )

        self._instruction_slice = find_string_in_tokens(self.instruction, toks, self.tokenizer)
        self._control_slice = find_string_in_tokens(self.control, toks, self.tokenizer) 
        self._assistant_role_slice = find_string_in_tokens(E_INST, toks, self.tokenizer)
        self._target_slice = find_string_in_tokens(self.target, toks, self.tokenizer) 
        self._loss_slice = slice(self._target_slice.start-1, self._target_slice.stop-1)
        assert self._target_slice.start == self._assistant_role_slice.stop, "Target starts when assistant role ends"

        self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')

    @property
    def instruction_str(self):
        return self.tokenizer.decode(self.input_ids[self._instruction_slice]).strip()

    @instruction_str.setter
    def instruction_str(self, instruction):
        self.instruction = instruction
        self._update_ids()
    
    @property
    def target_str(self):
        return self.tokenizer.decode(self.input_ids[self._target_slice]).strip()
    
    @target_str.setter
    def target_str(self, target):
        self.target = target
        self._update_ids()

    @property
    def control_str(self):
        return self.tokenizer.decode(self.input_ids[self._control_slice]).strip()
    
    @control_str.setter
    def control_str(self, control):
        self.control = control
        self._update_ids()
    
    @property
    def control_toks(self):
        return self.input_ids[self._control_slice]
    
    @control_toks.setter
    def control_toks(self, control_toks):
        self.control = self.tokenizer.decode(control_toks)
        self._update_ids()