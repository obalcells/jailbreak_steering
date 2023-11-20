import torch
import json
import gc
import os
import pandas as pd

def get_instructions_and_targets(csv_file: str, n_train_data: int=None):
    instructions = []
    targets = []

    dataset = pd.read_csv(csv_file)
    instructions = dataset['goal'].tolist()
    targets = dataset['target'].tolist()

    if n_train_data is not None:
        instructions = instructions[:n_train_data]
        targets = targets[:n_train_data]

    return instructions, targets

# check if a jailbreak for some prompt has already been computed
def check_if_done(file_path, instruction):
    if not os.path.exists(file_path):
        return False

    jailbreaks = json.load(open(file_path, "r"))

    for jailbreak in jailbreaks:
        if jailbreak["instruction"] == instruction:
            return True
        
    return False

def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)
    
# performs a binary search to look for a string inside some tokens, returns a slice of where the string's tokens are located
def find_string_in_tokens(string, tokens, tokenizer) -> slice:
    assert string in tokenizer.decode(tokens), "The string isn't contained in the whole array of tokens"
    # we first perform the binary search over the end index of the slice
    end_idx_left, end_idx_right = 0, len(tokens) 
    while end_idx_left != end_idx_right:
        mid = (end_idx_right + end_idx_left) // 2
        if string in tokenizer.decode(tokens[:mid]):
            end_idx_right = mid
        else:
            end_idx_left = mid + 1
    end_idx = end_idx_left
    # now we perform the binary search over the start index of the slice
    start_idx_left, start_idx_right = 0, end_idx-1 
    while start_idx_left != start_idx_right:
        mid = (start_idx_right + start_idx_left + 1) // 2
        if string in tokenizer.decode(tokens[mid:end_idx]):
            start_idx_left = mid
        else:
            start_idx_right = mid-1
    start_idx = start_idx_left
    string_slice = slice(start_idx, end_idx)
    assert string in tokenizer.decode(tokens[string_slice])
    return string_slice
