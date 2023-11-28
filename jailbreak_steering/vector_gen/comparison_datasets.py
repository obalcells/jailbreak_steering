# This file based off of https://github.com/nrimsky/SycophancySteering/blob/main/generate_vectors.py

import json
import torch as t

from torch.utils.data import Dataset

from jailbreak_steering.utils.tokenize_llama_chat import tokenize_llama_chat

class InstructionComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, tokenizer):
        self.data = json.load(open(data_path, "r")) 
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["instruction_inducing_behavior"]
        n_text = item["instruction_not_inducing_behavior"]
        p_tokens = self._instruction_to_tokens(p_text)
        n_tokens = self._instruction_to_tokens(n_text)
        return p_tokens, n_tokens

    def _instruction_to_tokens(self, instruction: str):
        instruction_toks = tokenize_llama_chat(self.tokenizer, [(instruction, None)], self.system_prompt, no_final_eos=True)
        return t.tensor(instruction_toks).unsqueeze(0)

class InstructionAnswerComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, tokenizer):
        self.data = json.load(open(data_path, "r")) 
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        i_text = item["instruction"]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        p_tokens = self._instruction_answer_to_tokens(i_text, p_text)
        n_tokens = self._instruction_answer_to_tokens(i_text, n_text)
        return p_tokens, n_tokens

    def _instruction_answer_to_tokens(self, instruction: str, answer: str):
        instruction_answer_toks = tokenize_llama_chat(self.tokenizer, [(instruction, answer)], self.system_prompt, no_final_eos=True)
        return t.tensor(instruction_answer_toks).unsqueeze(0)