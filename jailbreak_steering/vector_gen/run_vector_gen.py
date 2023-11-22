# This file based off of https://github.com/nrimsky/SycophancySteering/blob/main/generate_vectors.py

"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --label 'jailbreak_no_suffix' --data_path '/path/to/dataset' --layers 10 11 12 13 14 --save_activations
"""

import json
import torch as t
import argparse
import os

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List

from jailbreak_steering.utils.llama_wrapper import LlamaWrapper
from jailbreak_steering.utils.tokenize_llama_chat import tokenize_llama_chat
from jailbreak_steering.vector_gen.process_suffix_gen import NO_SUFFIX_DATASET_PATH, RANDOM_SUFFIX_DATASET_PATH

SAVE_VECTORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
SAVE_ACTIVATIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activations")
SYSTEM_PROMPT = None

DEFAILT_DATASET_PATH = NO_SUFFIX_DATASET_PATH
DEFAULT_LABEL = "jailbreak_no_suffix"

class InstructionComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, tokenizer):
        self.data = json.load(open(data_path, "r")) 
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_inst = item["instruction_inducing_behavior"]
        n_inst = item["instruction_not_inducing_behavior"]
        p_tokens = self._instruction_to_tokens(p_inst)
        n_tokens = self._instruction_to_tokens(n_inst)
        return p_tokens, n_tokens

    def _instruction_to_tokens(self, instruction: str):
        instruction_toks = tokenize_llama_chat(self.tokenizer, [(f"{instruction}", None)], self.system_prompt, no_final_eos=True)
        return t.tensor(instruction_toks).unsqueeze(0)

def generate_save_vectors(label: str, data_path: str, layers: List[int], save_activations: bool):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    """
    if not os.path.exists(SAVE_VECTORS_PATH):
        os.makedirs(SAVE_VECTORS_PATH)
    if save_activations and not os.path.exists(SAVE_ACTIVATIONS_PATH):
        os.makedirs(SAVE_ACTIVATIONS_PATH)

    model = LlamaWrapper(SYSTEM_PROMPT)
    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = InstructionComparisonDataset(
        data_path,
        SYSTEM_PROMPT,
        model.tokenizer,
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -1, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -1, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            os.path.join(
                SAVE_VECTORS_PATH,
                f"vec_layer_{make_tensor_save_suffix(layer, label)}.pt",
            ),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                os.path.join(
                    SAVE_ACTIVATIONS_PATH,
                    f"activations_pos_{make_tensor_save_suffix(layer, label)}.pt",
                ),
            )
            t.save(
                all_neg_layer,
                os.path.join(
                    SAVE_ACTIVATIONS_PATH,
                    f"activations_neg_{make_tensor_save_suffix(layer, label)}.pt",
                ),
            )

def make_tensor_save_suffix(layer, label):
    return f'{layer}_{label.split("/")[-1]}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL)
    parser.add_argument("--data_path", type=str, default=DEFAILT_DATASET_PATH)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)

    args = parser.parse_args()
    generate_save_vectors(args.label, args.data_path, args.layers, args.save_activations)